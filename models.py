import logging
from functools import partial
from autodp.transformer_zoo import ComposeGaussian, Composition
import torch
import torch.nn.functional as F
from torch.nn import SELU, ModuleList, Dropout, ReLU, Tanh, LazyLinear
from torch.optim import Adam, SGD
from torch_geometric.nn import BatchNorm
from args import support_args
from privacy import Calibrator, GaussianMechanism, NullMechanism, TopMFilter, supported_mechanisms
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout_fn, activation_fn, batchnorm, is_output_module):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout_fn
        self.activation = activation_fn
        self.is_output_module = is_output_module
        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] if num_layers > 0 else []
        self.layers = ModuleList([LazyLinear(out_channels) for out_channels in dimensions])
        self.bns = ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - int(is_output_module))]) if batchnorm else []
        self.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == self.num_layers - 1 and self.is_output_module:
                break

            x = self.bns[i](x) if self.bns else x
            x = self.dropout(x)
            x = self.activation(x)

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


class MultiStageClassifier(LightningModule):
    supported_combinations = {
        'cat', 'sum', 'max', 'mean' #, 'att'
    }

    def __init__(self, num_stages,
                 hidden_dim, output_dim,
                 pre_layers, post_layers, combination_type,
                 activation_fn, dropout_fn, batchnorm):

        super().__init__()
        self.combination_type = combination_type

        self.pre_mlps = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=pre_layers,
                dropout_fn=dropout_fn,
                activation_fn=activation_fn,
                batchnorm=batchnorm,
                is_output_module=False,
            )
            for _ in range(num_stages)
        ])

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation_fn=activation_fn,
            dropout_fn=dropout_fn,
            batchnorm=batchnorm,
            is_output_module=True
        )

    def forward(self, x_list):
        h_list = [mlp(x) for x, mlp in zip(x_list, self.pre_mlps)]
        h_combined = self.combine(h_list)
        h_out = self.post_mlp(h_combined)
        return F.log_softmax(h_out, dim=1)

    def combine(self, h_list):
        if self.combination_type == 'cat':
            return torch.cat(h_list, dim=-1)
        elif self.combination_type == 'sum':
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.combination_type == 'mean':
            return torch.stack(h_list, dim=0).mean(dim=0)
        elif self.combination_type == 'max':
            return torch.stack(h_list, dim=0).max(dim=0)
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
class PrivateNodeClassifier:
    supported_perturbations = {'aggr', 'feature', 'graph'}

    supported_activations = {
        'relu': partial(ReLU, inplace=True),
        'selu': partial(SELU, inplace=True),
        'tanh': Tanh,
    }

    def __init__(self,
                 num_classes,
                 perturbation:  dict(help='perturbation method', option='-p', choices=supported_perturbations) = 'aggr',
                 mechanism:     dict(help='perturbation mechanism', choices=supported_mechanisms) = 'gaussian',
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
                 precision:     dict(help='precision of the model', choices=[16, 32]) = 32,
                 ):

        super().__init__()

        self.hops = hops
        self.encoder_layers = encoder_layers
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.accelerator = 'cpu' if cpu else 'gpu'
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.precision = precision
        self.perturbation = perturbation

        if perturbation == 'graph':
            self.base_mechanism = TopMFilter(noise_scale=0.0)
        else:
            self.base_mechanism = supported_mechanisms[mechanism](noise_scale=0.0)

        activation_fn = self.supported_activations[activation]()
        dropout_fn = Dropout(dropout, inplace=True)

        self.encoder_classifier = MultiStageClassifier(
            num_stages=1,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            pre_layers=encoder_layers,
            post_layers=1,
            combination_type='cat',
            activation_fn=activation_fn,
            dropout_fn=dropout_fn,
            batchnorm=batchnorm
        )

        self.multi_stage_classifier = MultiStageClassifier(
            num_stages=hops+1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=pre_layers,
            post_layers=post_layers,
            combination_type=combine,
            activation_fn=activation_fn,
            dropout_fn=dropout_fn,
            batchnorm=batchnorm
        )

        self.accuracy = {
            'train': Accuracy(),
            'val': Accuracy(),
            'test': Accuracy(),
        }

        self.setup_models()
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder_classifier.reset_parameters()
        self.multi_stage_classifier.reset_parameters()
        for stage in self.accuracy:
            self.accuracy[stage].reset()

    def setup_models(self):
        for model in [self.encoder_classifier, self.multi_stage_classifier]:
            model.configure_optimizers = partial(self.configure_optimizers, model=model)
            model.training_step = partial(self.step, stage='train')
            model.validation_step = partial(self.step, stage='val')
            model.test_step = partial(self.step, stage='test')

    def configure_optimizers(self, model):
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def fit(self, data):
        self.data = data
        self.reset_parameters()

        ### pre-training encoder ###

        if self.encoder_layers > 0:
            logging.info('Pre-training encoder...')
            self._pre_train_flag = True
            
            checkpoint_callback = ModelCheckpoint(
                monitor="val/loss",
                dirpath='./checkpoints/',
                save_last=False,
                save_top_k=1,
                mode='min',
                save_weights_only=True,
            )

            trainer = Trainer(
                enable_checkpointing=True,
                callbacks=[checkpoint_callback],
                devices='auto',
                accelerator=self.accelerator,
                logger=False,
                max_epochs=self.pre_epochs,
                precision=self.precision,
                enable_model_summary=False,
            )

            trainer.fit(
                model=self.encoder_classifier, 
                train_dataloader=self.train_dataloader(), 
                val_dataloaders=self.val_dataloader()
            )

        ### precompute cached data ###
        logging.info('Precomputing cached data...')

        if self.perturbation == 'graph':
            data.adj_t = self.base_mechanism.perturb(data.adj_t, num_nodes=data.num_nodes)

        h = self.encoder_classifier.encode([data.x])

        # TODO: make sure encode in trained L2-normalized
        if self.perturbation in {'aggr', 'feature'}:
            h = self.base_mechanism.normalize(h)

        data.x_list = [h]

        for _ in range(1, self.hops + 1):
            h = data.x_list[-1]

            if self.perturbation == 'feature':
                h = self.base_mechanism.perturb(h, sensitivity=1)

            h = data.adj_t.matmul(data.x_list[-1])

            if self.perturbation == 'aggr':
                h = self.base_mechanism.perturb(h, sensitivity=1)

            if self.perturbation in {'aggr', 'feature'}:
                h = self.base_mechanism.normalize(h)

            data.x_list.append(h)

        ### training ###
        logging.info('Training...')

        self._pre_train_flag = False
        for stage in self.accuracy:
            self.accuracy[stage].reset()

        checkpoint_callback = ModelCheckpoint(
            monitor="val/acc",
            dirpath='./checkpoints/',
            save_last=False,
            save_top_k=1,
            mode='max',
            save_weights_only=True,
        )

        trainer = Trainer(
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
            devices='auto',
            accelerator=self.accelerator,
            logger=False,
            max_epochs=self.pre_epochs,
            precision=self.precision,
            enable_model_summary=False,
        )

        trainer.fit(
            model=self.multi_stage_classifier, 
            train_dataloader=self.train_dataloader(), 
            val_dataloaders=self.val_dataloader()
        )

        metrics = trainer.test(dataloaders=self.test_dataloader(), ckpt_path='best', verbose=False)
        return metrics[0]

    def train_dataloader(self):
        train_idx = self.data.train_mask.nonzero(as_tuple=False).view(-1)
        train_loader = DataLoader(train_idx, batch_size=1024, shuffle=True, num_workers=6, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_idx = self.data.val_mask.nonzero(as_tuple=False).view(-1)
        val_loader = DataLoader(val_idx, batch_size=1024, shuffle=True, num_workers=6, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        test_idx = self.data.test_mask.nonzero(as_tuple=False).view(-1)
        test_loader = DataLoader(test_idx, batch_size=1024, shuffle=True, num_workers=6, pin_memory=True)
        return test_loader

    def step(self, stage, batch, _):
        if self._pre_train_flag:
            model = self.encoder_classifier
            y_pred = model([self.data.x[batch]])
        else:
            model = self.multi_stage_classifier
            y_pred = model([x[batch] for x in self.data.x_list])

        y_true = self.data.y[batch]
        acc = self.accuracy[stage](y_pred, y_true)
        metrics = {f'{stage}/acc': acc}

        if stage != 'test':
            loss = F.nll_loss(input=y_pred, target=y_true)
            metrics[f'{stage}/loss'] = loss

        model.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss if stage == 'train' else metrics

    def build_mechanism(self, noise_scale=None):
        if noise_scale is None:
            noise_scale = self.base_mechanism.noise_scale

        if self.hops == 0 or noise_scale == 0.0:
            return NullMechanism()

        self.base_mechanism.update(noise_scale=noise_scale)

        if self.perturbation == 'graph':
            return self.base_mechanism
        else:
            compose = ComposeGaussian() if isinstance(self.base_mechanism, GaussianMechanism) else Composition()
            composed_mech = compose([self.base_mechanism], [self.hops])
            return composed_mech


    def calibrate(self, epsilon, delta):
        mechanism_builder = lambda noise_scale: self.build_mechanism(noise_scale=noise_scale)
        noise_scale = Calibrator(mechanism_builder).calibrate(eps=epsilon, delta=delta)
        self.base_mechanism.update(noise_scale=noise_scale)
        return self
