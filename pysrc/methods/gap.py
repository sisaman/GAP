import torch
import numpy as np
import logging
from time import time
from torch.nn import Module
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from pysrc.console import console
from pysrc.methods.base import MethodBase
from pysrc.trainer import Trainer
from pysrc.data.loader.poisson import PoissonDataLoader
from pysrc.privacy.mechanisms import ComposedNoisyMechanism
from pysrc.privacy.algorithms import NoisySGD, PMA
from pysrc.classifiers import MultiMLPClassifier
from pysrc.data.transforms import NeighborSampler
from pysrc.trainer.typing import Metrics, TrainerStage


class GAP (MethodBase):
    supported_dp_levels = {'edge', 'node'}
    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 dp_level:      dict(help='level of privacy protection', option='-l', choices=supported_dp_levels) = 'edge',
                 epsilon:       dict(help='DP epsilon parameter', option='-e') = np.inf,
                 delta:         dict(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d') = 'auto',
                 hops:          dict(help='number of hops', option='-k') = 2,
                 max_degree:    dict(help='max degree to sample per each node (if 0, disables degree sampling)') = 0,
                 hidden_dim:    dict(help='dimension of the hidden layers') = 16,
                 encoder_layers:dict(help='number of encoder MLP layers') = 2,
                 pre_layers:    dict(help='number of pre-combination MLP layers') = 1,
                 post_layers:   dict(help='number of post-combination MLP layers') = 1,
                 combine:       dict(help='combination type of transformed hops', choices=MultiMLPClassifier.supported_combinations) = 'cat',
                 activation:    dict(help='type of activation function', choices=supported_activations) = 'selu',
                 dropout:       dict(help='dropout rate') = 0.0,
                 batch_norm:    dict(help='if true, then model uses batch normalization') = True,
                 optimizer:     dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
                 learning_rate: dict(help='learning rate', option='--lr') = 0.01,
                 weight_decay:  dict(help='weight decay (L2 penalty)') = 0.0,
                 cpu:           dict(help='if true, then model is trained on CPU') = False,
                 pre_epochs:    dict(help='number of epochs for pre-training (ignored if encoder_layers=0)') = 100,
                 epochs:        dict(help='number of epochs for training') = 100,
                 batch_size:    dict(help='batch size (if 0, performs full-batch training)') = 0,
                 max_grad_norm: dict(help='maximum norm of the per-sample gradients (ignored if dp_level=edge)') = 1.0,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 ):

        assert not (dp_level == 'node' and epsilon < np.inf and hops > 0 and max_degree <= 0), 'max_degree must be positive for node-level DP'
        assert not (dp_level == 'node' and epsilon < np.inf and batch_size <= 0), 'batch_size must be positive for node-level DP'

        if dp_level == 'node' and epsilon < np.inf and batch_norm:
            logging.warn('batch normalization is not supported for node-level DP, setting batch_norm to False')
            batch_norm = False

        if encoder_layers == 0 and pre_epochs > 0:
            logging.info('encoder is not available, setting pre_epochs to 0')
            pre_epochs = 0

        self.dp_level = dp_level
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
        self.noise_scale = 0.0 # used to save noise calibration results
        activation_fn = self.supported_activations[activation]

        self.encoder = MultiMLPClassifier(
            num_inputs=1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=encoder_layers,
            post_layers=1,
            combination='cat',
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.classifier = MultiMLPClassifier(
            num_inputs=hops+1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=pre_layers,
            post_layers=post_layers,
            combination=combine,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def init_privacy_mechanisms(self):
        self.pma_mechanism = PMA(noise_scale=self.noise_scale, hops=self.hops)

        if self.dp_level == 'edge':
            mechanism_list = [self.pma_mechanism]
        elif self.dp_level == 'node':
            dataset = self.data_loader('train').dataset
            dataset_size = len(dataset)
            batch_size = len(dataset) if self.batch_size <= 0 else self.batch_size

            self.pretraining_noisy_sgd = NoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=dataset_size, 
                batch_size=batch_size, 
                epochs=self.pre_epochs,
                max_grad_norm=self.max_grad_norm,
            )

            self.training_noisy_sgd = NoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=dataset_size, 
                batch_size=batch_size, 
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
            )

            mechanism_list = [self.pretraining_noisy_sgd, self.pma_mechanism, self.training_noisy_sgd]

        composed_mech = ComposedNoisyMechanism(
            noise_scale=self.noise_scale,
            mechanism_list=mechanism_list, 
            coeff_list=[1]*len(mechanism_list)
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                if np.isinf(self.epsilon):
                    self.delta = 0.0
                else:
                    data_size = self.data.num_edges if self.dp_level == 'edge' else self.data.num_nodes
                    self.delta = 1. / (10 ** len(str(data_size)))
                logging.info('delta = %.0e', self.delta)
            self.noise_scale = composed_mech.calibrate(eps=self.epsilon, delta=self.delta)
            logging.info(f'noise scale: {self.noise_scale:.4f}\n')

    def fit(self, data: Data) -> Metrics:
        self.data = data
        
        with console.status(f'moving data to {self.device}'):
            self.data.to(self.device)

        start_time = time()
        self.init_privacy_mechanisms()

        logging.info('step 1: encoder module')
        self.pretrain_encoder()

        logging.info('step 2: aggregation module')
        self.precompute_aggregations()

        logging.info('step 3: classification module')
        metrics = self.train_classifier()
        
        end_time = time()
        metrics['time'] = end_time - start_time

        return metrics

    def pretrain_encoder(self):
        if self.encoder_layers > 0:
            self.encoder.to(self.device)
            self.data.x = torch.stack([self.data.x], dim=-1)

            trainer = Trainer(
                epochs=self.pre_epochs, 
                use_amp=self.use_amp, 
                monitor='val/acc', monitor_mode='max', 
                device=self.device,
                noisy_sgd=self.pretraining_noisy_sgd if self.dp_level == 'node' else None,
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
            self.encoder.to('cpu')

    def precompute_aggregations(self):
        if self.dp_level == 'node' and self.hops > 0 and self.max_degree > 0:
            with console.status('bounding the number of neighbors per node'):
                self.data = NeighborSampler(self.max_degree)(self.data)

        sensitivity = 1 if self.dp_level == 'edge' else np.sqrt(self.max_degree)
        with console.status('computing aggregations'):
            self.data = self.pma_mechanism(self.data, sensitivity=sensitivity)
        self.data.to('cpu', 'adj_t')

    def train_classifier(self) -> Metrics:
        self.classifier.to(self.device)

        trainer = Trainer(
            epochs=self.epochs, 
            use_amp=self.use_amp, 
            monitor='val/acc', monitor_mode='max', 
            device=self.device,
            noisy_sgd=self.training_noisy_sgd if self.dp_level == 'node' else None,
        )

        metrics = trainer.fit(
            model=self.classifier, 
            optimizer=self.configure_optimizers(self.classifier),
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=self.data_loader('test'),
            checkpoint=False,
        )

        self.classifier.to('cpu')
        return metrics

    def data_loader(self, stage: TrainerStage) -> PoissonDataLoader:
        mask = self.data[f'{stage}_mask']
        x = self.data.x[mask]
        y = self.data.y[mask]
        dataset = TensorDataset(x, y)
        batch_size = len(dataset) if self.batch_size <= 0 else self.batch_size
        return PoissonDataLoader(dataset, batch_size=batch_size)

    def configure_optimizers(self, model: Module) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
