from console import console
import logging
from functools import partial
from autodp.transformer_zoo import ComposeGaussian
import torch
import torch.nn.functional as F
from torch.nn import SELU, ModuleList, Dropout, ReLU, Tanh, LazyLinear, Module, ModuleDict, LazyBatchNorm1d
from torch.optim import Adam, SGD
from args import support_args
from privacy import Calibrator, GaussianMechanism, NullMechanism, TopMFilter
from torchmetrics import Accuracy, MeanMetric
from torch.utils.data import DataLoader
from trainer import Trainer

supported_activations = {
    'relu': partial(ReLU, inplace=True),
    'selu': partial(SELU, inplace=True),
    'tanh': Tanh,
}

class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, activation, batchnorm):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers = ModuleList([LazyLinear(dim) for dim in dimensions])
        
        num_bns = batchnorm * (num_layers - 1)
        self.bns = batchnorm and ModuleList([LazyBatchNorm1d() for _ in range(num_bns)])
        
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

    def __init__(self, num_stages, hidden_dim, output_dim, pre_layers, post_layers, combination_type, activation, dropout, batchnorm):

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

    def forward(self, x_list):
        h_list = [mlp(x) for x, mlp in zip(x_list, self.pre_mlps)]
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
    def encode(self, x_list):
        self.eval()
        h_list = [mlp(x) for x, mlp in zip(x_list, self.pre_mlps)]
        h_combined = self.combine(h_list)
        return h_combined

    def reset_parameters(self):
        for mlp in self.pre_mlps:
            mlp.reset_parameters()
        self.post_mlp.reset_parameters()


@support_args
class GAP(Module):
    supported_perturbations = {'aggr', 'graph'}

    # TODO add node-level DP params
    def __init__(self,
                 num_classes,
                 perturbation:  dict(help='perturbation method', option='-p', choices=supported_perturbations) = 'aggr',
                 hops:          dict(help='number of hops', option='-k') = 1,
                 hidden_dim:    dict(help='dimension of the hidden layers') = 16,
                 encoder_layers:dict(help='number of encoder MLP layers') = 2,
                 pre_layers:    dict(help='number of pre-combination MLP layers') = 1,
                 post_layers:   dict(help='number of post-combination MLP layers') = 1,
                 combine:       dict(help='combination type of transformed hops', choices=MultiStageClassifier.supported_combinations) = 'cat',
                 activation:    dict(help='type of activation function', choices=supported_activations) = 'relu',
                 dropout:       dict(help='dropout rate (between zero and one)') = 0.0,
                 batchnorm:     dict(help='if True, then model uses batch normalization') = True,
                 optimizer:     dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
                 learning_rate: dict(help='learning rate', option='--lr') = 0.01,
                 weight_decay:  dict(help='weight decay (L2 penalty)') = 0.0,
                 cpu:           dict(help='if True, then model is trained on CPU') = False,
                 pre_epochs:    dict(help='number of epochs for pre-training') = 100,
                 epochs:        dict(help='number of epochs for training') = 100,
                 batch_size:    dict(help='batch size (if zero, performs full-batch training)') = 0,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 ):

        super().__init__()
        self.hops = hops
        self.encoder_layers = encoder_layers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.device = 'cpu' if cpu else 'cuda'
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.perturbation = perturbation

        if perturbation == 'graph':
            self.base_mechanism = TopMFilter(noise_scale=0.0)
        else:
            self.base_mechanism = GaussianMechanism(noise_scale=0.0)

        self.encoder = MultiStageClassifier(
            num_stages=1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=encoder_layers,
            post_layers=1,
            combination_type='cat',
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm
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
            batchnorm=batchnorm
        )

        self.metrics = ModuleDict({
            'train/loss': MeanMetric(compute_on_step=False),
            'train/acc': Accuracy(compute_on_step=False),
            'val/loss': MeanMetric(compute_on_step=False),
            'val/acc': Accuracy(compute_on_step=False),
            'test/acc': Accuracy(compute_on_step=False),
        })

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
        for stage in self.metrics:
            self.metrics[stage].reset()

    def configure_optimizers(self):
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        parameters = self.encoder.parameters() if self._pre_train_flag else self.classifier.parameters()
        return Optim(parameters, lr=self.learning_rate, weight_decay=self.weight_decay)

    def set_training_state(self, pre_train):
        self._pre_train_flag = pre_train
        
        for metric in self.metrics:
            self.metrics[metric].reset()

    def fit(self, data):
        self.data = data

        ### pretraining encoder module ###
        if self.encoder_layers > 0:
            logging.info('pretraining encoder module...')
            self.pretrain_encoder()

        ### precompute aggregation module ###
        with console.status('precomputing aggregation module...'):
            self.precompute_aggregations()

        ### training classification module ###
        logging.info('training classification module...')
        return self.train_model()

    def pretrain_encoder(self):
        self.set_training_state(pre_train=True)
        
        trainer = Trainer(
            epochs=self.pre_epochs, 
            use_amp=self.use_amp, 
            monitor='val/loss', monitor_mode='min', 
            device=self.device
        )

        trainer.fit(
            model=self, 
            train_dataloader=self.train_dataloader(), 
            val_dataloader=self.val_dataloader(),
            test_dataloader=self.test_dataloader(),
            checkpoint=True
        )

        self.encoder = trainer.load_best_model().encoder

    @torch.no_grad()
    def precompute_aggregations(self):
        data = self.data

        if self.perturbation == 'graph':
            data.adj_t = self.base_mechanism.perturb(data.adj_t, num_nodes=data.num_nodes)

        h = self.encoder.encode([data.x])

        if self.perturbation == 'aggr':
            h = self.base_mechanism.normalize(h)

        data.x_list = [h]

        for _ in range(1, self.hops + 1):

            # aggregate
            h = data.adj_t.matmul(h)

            if self.perturbation == 'aggr':

                # perturb
                h = self.base_mechanism.perturb(h, sensitivity=1)

                # normalize
                h = self.base_mechanism.normalize(h)    

            data.x_list.append(h)

    def train_model(self):
        self.set_training_state(pre_train=False)

        trainer = Trainer(
            epochs=self.epochs, 
            use_amp=self.use_amp, 
            monitor='val/loss', monitor_mode='min', 
            device=self.device,
        )

        metrics = trainer.fit(
            model=self, 
            train_dataloader=self.train_dataloader(), 
            val_dataloader=self.val_dataloader(),
            test_dataloader=self.test_dataloader(),
            checkpoint=False,
        )

        return metrics

    def train_dataloader(self):
        train_idx = self.data.train_mask.nonzero(as_tuple=False).view(-1)
        return self.index_loader(train_idx)


    def val_dataloader(self):
        val_idx = self.data.val_mask.nonzero(as_tuple=False).view(-1)
        return self.index_loader(val_idx)

    def test_dataloader(self):
        test_idx = self.data.test_mask.nonzero(as_tuple=False).view(-1)
        return self.index_loader(test_idx)

    def index_loader(self, idx):        
        if self.batch_size <= 0:
            return [idx]
        else:
            return DataLoader(idx, batch_size=self.batch_size, shuffle=True)

    def aggregate_metrics(self, stage='train'):
        metrics = {}

        for metric_name, metric_value in self.metrics.items():
            if metric_name.startswith(stage):
                value = metric_value.compute()
                metric_value.reset()
                if torch.is_tensor(value):
                    value = value.item()
                value *= 100 if metric_name.endswith('acc') else 1
                metrics[metric_name] = value

        return metrics

    def step(self, batch, stage):

        if self._pre_train_flag:
            preds = self.encoder([self.data.x[batch]])
        else:
            preds = self.classifier([x[batch] for x in self.data.x_list])

        target = self.data.y[batch]
        self.metrics[f'{stage}/acc'].update(preds=preds, target=target)

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=target)
            self.metrics[f'{stage}/loss'].update(loss, weight=len(batch))

        return loss

    def build_mechanism(self, noise_scale=None):
        if noise_scale is None:
            noise_scale = self.base_mechanism.noise_scale

        if self.hops == 0 or noise_scale == 0.0:
            return NullMechanism()

        self.base_mechanism.update(noise_scale=noise_scale)

        if self.perturbation == 'graph':
            return self.base_mechanism
        else:
            compose = ComposeGaussian()
            composed_mech = compose([self.base_mechanism], [self.hops])
            return composed_mech


    def calibrate(self, epsilon, delta):
        mechanism_builder = lambda noise_scale: self.build_mechanism(noise_scale=noise_scale)
        noise_scale = Calibrator(mechanism_builder).calibrate(eps=epsilon, delta=delta)
        logging.info(f'noise scale: {noise_scale:.4f}\n')
        self.base_mechanism.update(noise_scale=noise_scale)
        return self
