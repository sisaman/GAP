from console import console
import numpy as np
import logging
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import SELU, ModuleList, Dropout, ReLU, Tanh, LazyLinear, Module, LazyBatchNorm1d
from torch.optim import Adam, SGD
from args import support_args
from privacy import TopMFilter, NoisySGD, PMA, ComposedNoisyMechanism
from torch.utils.data import DataLoader, TensorDataset
from trainer import Trainer
from datasets import NeighborSampler
from opacus.grad_sample import register_grad_sampler


supported_activations = {
    'relu': partial(ReLU, inplace=True),
    'selu': partial(SELU, inplace=True),
    'tanh': Tanh,
}


@register_grad_sampler(LazyLinear)
def compute_lazy_linear_grad_sample(layer, activations, backprops):
    gs = torch.einsum("n...i,n...j->nij", backprops, activations)
    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, activation, batchnorm):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers = ModuleList([LazyLinear(dim) for dim in dimensions])
        
        num_bns = batchnorm * (num_layers - 1)
        self.bns = ModuleList([LazyBatchNorm1d() for _ in range(num_bns)]) if batchnorm else []
        
        self.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < self.num_layers - 1:
                x = self.bns[i](x) if self.bns else x
                x = self.dropout(x)
                x = self.activation(x)

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
        for bn in self.bns:
            bn.reset_parameters()


class MultiStageClassifier(Module):
    supported_combinations = {
        'cat', 'sum', 'max', 'mean', 'att'   ### TODO implement attention
    }

    def __init__(self, num_stages, hidden_dim, output_dim, pre_layers, post_layers, 
                 combination_type, activation, dropout, batchnorm):

        super().__init__()
        self.combination_type = combination_type

        self.pre_mlps = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=pre_layers,
                dropout=dropout,
                activation=activation,
                batchnorm=batchnorm,
            )] * num_stages
        )

        self.bn = LazyBatchNorm1d() if batchnorm else False
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
        )

    def forward(self, x_stack):
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.pre_mlps)]
        h = self.combine(h_list)
        h = F.normalize(h, p=2, dim=-1)
        h = self.bn(h) if self.bn else h
        h = self.dropout(h)
        h = self.activation(h)
        h = self.post_mlp(h)
        return F.log_softmax(h, dim=-1)

    def combine(self, h_list):
        if self.combination_type == 'cat':
            return torch.cat(h_list, dim=-1)
        elif self.combination_type == 'sum':
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.combination_type == 'max':
            return torch.stack(h_list, dim=0).max(dim=0).values
        elif self.combination_type == 'att':
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown combination type {self.combination_type}')

    @torch.no_grad()
    def encode(self, x_stack):
        self.eval()
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.pre_mlps)]
        h_combined = self.combine(h_list)
        return h_combined

    def step(self, batch, stage):
        x_stack, y = batch
        preds = self(x_stack)
        acc = (preds.argmax(dim=1) == y).float().mean() * 100
        metrics = {f'{stage}/acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics[f'{stage}/loss'] = loss.detach()

        return loss, metrics

    def reset_parameters(self):
        if self.bn:
            self.bn.reset_parameters()

        for mlp in self.pre_mlps:
            mlp.reset_parameters()
        
        self.post_mlp.reset_parameters()


@support_args
class GAP:
    supported_dp_levels = {'edge', 'node'}
    supported_perturbations = {'aggr', 'graph'}

    def __init__(self,
                 num_classes,
                 dp_level:      dict(help='level of privacy protection', choices=supported_dp_levels) = 'edge',
                 perturbation:  dict(help='perturbation method', option='-p', choices=supported_perturbations) = 'aggr',
                 epsilon:       dict(help='DP epsilon parameter', option='-e') = np.inf,
                 delta:         dict(help='DP delta parameter', option='-d') = 1e-6,
                 hops:          dict(help='number of hops', option='-k') = 2,
                 max_degree:    dict(help='max degree per each node') = 0,
                 hidden_dim:    dict(help='dimension of the hidden layers') = 16,
                 encoder_layers:dict(help='number of encoder MLP layers') = 2,
                 pre_layers:    dict(help='number of pre-combination MLP layers') = 1,
                 post_layers:   dict(help='number of post-combination MLP layers') = 1,
                 combine:       dict(help='combination type of transformed hops', choices=MultiStageClassifier.supported_combinations) = 'cat',
                 activation:    dict(help='type of activation function', choices=supported_activations) = 'selu',
                 dropout:       dict(help='dropout rate (between zero and one)') = 0.0,
                 batchnorm:     dict(help='if True, then model uses batch normalization') = True,
                 optimizer:     dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
                 learning_rate: dict(help='learning rate', option='--lr') = 0.01,
                 weight_decay:  dict(help='weight decay (L2 penalty)') = 0.0,
                 cpu:           dict(help='if True, then model is trained on CPU') = False,
                 pre_epochs:    dict(help='number of epochs for pre-training') = 100,
                 epochs:        dict(help='number of epochs for training') = 100,
                 batch_size:    dict(help='batch size (if zero, performs full-batch training)') = 0,
                 max_grad_norm: dict(help='maximum norm of the per-sample gradients (ignored when using edge-level DP)') = 1.0,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 ):

        super().__init__()

        assert dp_level == 'edge' or perturbation == 'aggr'
        assert dp_level == 'edge' or max_degree > 0 
        assert dp_level == 'edge' or batch_size > 0
        assert encoder_layers == 0 or pre_epochs > 0

        self.dp_level = dp_level
        self.perturbation = perturbation
        self.epsilon = epsilon
        self.delta = delta
        self.hops = hops
        self.max_degree = max_degree
        self.encoder_layers = encoder_layers
        self.device = 'cpu' if cpu else 'cuda'
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.noise_scale = 1.0 # will be calibrated later

        self.encoder = MultiStageClassifier(
            num_stages=1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=encoder_layers,
            post_layers=1,
            combination_type='cat',
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        self.classifier = MultiStageClassifier(
            num_stages=hops+1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=pre_layers,
            post_layers=post_layers,
            combination_type=combine,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def set_training_state(self, pre_train):
        self._pre_train_flag = pre_train

    def init_privacy_mechanisms(self):
        self.pma_mechanism = PMA(noise_scale=self.noise_scale, hops=self.hops)

        if self.perturbation == 'graph':
            self.graph_mechanism = TopMFilter(noise_scale=self.noise_scale)
            mechanism_list = [self.graph_mechanism]
        elif self.dp_level == 'edge':
            mechanism_list = [self.pma_mechanism]
        elif self.dp_level == 'node':
            dataset_size = len(self.data_loader('train').dataset)

            self.pretraining_noisy_sgd = NoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=dataset_size, 
                batch_size=self.batch_size, 
                epochs=self.pre_epochs,
                max_grad_norm=self.max_grad_norm,
            )

            self.training_noisy_sgd = NoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=dataset_size, 
                batch_size=self.batch_size, 
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
            )

            mechanism_list = [self.pretraining_noisy_sgd, self.pma_mechanism, self.training_noisy_sgd]

        composed_mech = ComposedNoisyMechanism(
            noise_scale=self.noise_scale, 
            mechanism_list=mechanism_list, 
            coeff_list=[1]*len(mechanism_list)
        )

        with console.status('calibrating noise to privacy budget...'):
            noise_scale = composed_mech.calibrate(eps=self.epsilon, delta=self.delta)
            logging.info(f'noise scale: {noise_scale:.4f}\n')

    def fit(self, data):
        self.data = NeighborSampler(self.max_degree)(data)

        self.init_privacy_mechanisms()

        if self.encoder_layers > 0:
            logging.info('pretraining encoder module...')
            self.pretrain_encoder()

        with console.status('precomputing aggregation module...'):
            self.precompute_aggregations()

        logging.info('training classification module...')
        return self.train_classifier()

    def pretrain_encoder(self):
        self.set_training_state(pre_train=True)
        self.data.x = torch.stack([self.data.x], dim=-1)

        trainer = Trainer(
            epochs=self.pre_epochs, 
            use_amp=self.use_amp, 
            monitor='val/loss', monitor_mode='min', 
            device=self.device,
            dp_mechanism=self.pretraining_noisy_sgd if self.dp_level == 'node' else None,
        )

        trainer.fit(
            model=self.encoder,
            optimizer=self.configure_optimizers(self.encoder), 
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=None,
            checkpoint=True
        )

        self.encoder = trainer.load_best_model()
        self.data.x = self.encoder.encode(self.data.x)

    def precompute_aggregations(self):
        if self.perturbation == 'graph':
            self.data = self.graph_mechanism(self.data)

        sensitivity = 1 if self.dp_level == 'edge' else np.sqrt(self.max_degree)
        self.data = self.pma_mechanism(self.data, sensitivity=sensitivity)

    def train_classifier(self):
        self.set_training_state(pre_train=False)

        trainer = Trainer(
            epochs=self.epochs, 
            use_amp=self.use_amp, 
            monitor='val/loss', monitor_mode='min', 
            device=self.device,
            dp_mechanism=self.training_noisy_sgd if self.dp_level == 'node' else None,
        )

        metrics = trainer.fit(
            model=self.classifier, 
            optimizer=self.configure_optimizers(self.classifier),
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=self.data_loader('test'),
            checkpoint=False,
        )

        return metrics

    def data_loader(self, stage):
        mask = self.data[f'{stage}_mask']
        x = self.data.x[mask]
        y = self.data.y[mask]

        if self.batch_size == 0:
            return TensorDataset(x.unsqueeze(0), y.unsqueeze(0))
        else:
            dataset = TensorDataset(x, y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self, model):
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
