import torch
from pysrc.console import console
import numpy as np
import logging
from typing import Annotated, Union
from time import time
from torch.nn import Module
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from pysrc.methods.base import MethodBase
from pysrc.trainer import Trainer
from pysrc.privacy.algorithms import AsymmetricRandResponse
from pysrc.privacy.algorithms import GNNBasedNoisySGD
from pysrc.privacy.mechanisms import GaussianMechanism
from pysrc.classifiers import GraphSAGEClassifier
from pysrc.privacy.mechanisms import ComposedNoisyMechanism
from pysrc.data.transforms import NeighborSampler
from pysrc.trainer.typing import Metrics, TrainerStage


class GraphSAGE (MethodBase):
    supported_dp_levels = {'edge', 'node'}
    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 dp_level:      Annotated[str,   dict(help='level of privacy protection', option='-l', choices=supported_dp_levels)] = 'edge',
                 epsilon:       Annotated[float, dict(help='DP epsilon parameter', option='-e')] = np.inf,
                 delta:         Annotated[Union[str, float], dict(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   dict(help='max degree to sample per each node (if 0, disables degree sampling)')] = 0,
                 hidden_dim:    Annotated[int,   dict(help='dimension of the hidden layers')] = 16,
                 encoder_layers:Annotated[int,   dict(help='number of encoder MLP layers')] = 2,
                 mp_layers:     Annotated[int,   dict(help='number of GNN layers')] = 1,
                 post_layers:   Annotated[int,   dict(help='number of post-processing MLP layers')] = 1,
                 activation:    Annotated[str,   dict(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:       Annotated[float, dict(help='dropout rate')] = 0.0,
                 batch_norm:    Annotated[bool,  dict(help='if true, then model uses batch normalization')] = True,
                 optimizer:     Annotated[str,   dict(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate: Annotated[float, dict(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:  Annotated[float, dict(help='weight decay (L2 penalty)')] = 0.0,
                 cpu:           Annotated[bool,  dict(help='if true, then model is trained on CPU')] = False,
                 epochs:        Annotated[int,   dict(help='number of epochs for training')] = 100,
                 batch_size:    Annotated[int,   dict(help='batch size (if 0, performs full-batch training)')] = 0,
                 max_grad_norm: Annotated[float, dict(help='maximum norm of the per-sample gradients (ignored if dp_level=edge)')] = 1.0,
                 use_amp:       Annotated[bool,  dict(help='use automatic mixed precision training')] = False,
                 ):

        assert mp_layers >= 1, 'number of message-passing layers must be at least 1'
        assert not (dp_level == 'node' and epsilon < np.inf and mp_layers > 1), 'node-level DP is not supported for more than one message-passing layer'
        assert not (dp_level == 'node' and epsilon < np.inf and max_degree <= 0), 'max_degree must be positive for node-level DP'
        assert not (dp_level == 'node' and epsilon < np.inf and batch_size <= 0), 'batch_size must be positive for node-level DP'

        if dp_level == 'node' and batch_norm:
            logging.warn('batch normalization is not supported for node-level DP, setting it to False')
            batch_norm = False

        self.dp_level = dp_level
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.encoder_layers = encoder_layers
        self.mp_layers = mp_layers
        self.post_layers = post_layers
        self.device = 'cpu' if cpu else 'cuda'
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.noise_scale = 0.0 # used to save noise calibration results
        activation_fn = self.supported_activations[activation]

        self.classifier = GraphSAGEClassifier(
            hidden_dim=hidden_dim, 
            output_dim=num_classes, 
            pre_layers=encoder_layers,
            mp_layers=mp_layers, 
            post_layers=post_layers, 
            normalize=dp_level == 'node',
            activation_fn=activation_fn, 
            dropout=dropout, 
            batch_norm=batch_norm,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.classifier.reset_parameters()
        if hasattr(self, 'noisy_aggr_hook'):
            self.noisy_aggr_hook.remove()

    def init_privacy_mechanisms(self):
        if self.dp_level == 'edge':
            mech = AsymmetricRandResponse(eps=self.epsilon)
            self.graph_mechanism = mech
        else:
            self.training_noisy_sgd = GNNBasedNoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=self.data.train_mask.sum().item(),
                batch_size=self.batch_size, 
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
                max_degree=self.max_degree,
            )
            self.noisy_aggr_gm = GaussianMechanism(noise_scale=self.noise_scale)
            mech = ComposedNoisyMechanism(
                noise_scale=self.noise_scale,
                mechanism_list=[self.training_noisy_sgd, self.noisy_aggr_gm], 
                coeff_list=[1,1]
            )
            self.noisy_aggr_hook = self.classifier.gnn.convs[0].register_message_and_aggregate_forward_hook(
                lambda module, inputs, output: 
                    self.noisy_aggr_gm(data=output, sensitivity=np.sqrt(self.max_degree)) if not module.training else output
            )

            with console.status('calibrating noise to privacy budget'):
                if self.delta == 'auto':
                    if np.isinf(self.epsilon):
                        self.delta = 0.0
                    else:
                        self.delta = 1. / (10 ** len(str(self.data.num_edges)))
                    logging.info('delta = %.0e', self.delta)

                self.noise_scale = mech.calibrate(eps=self.epsilon, delta=float(self.delta))
                logging.info(f'noise scale: {self.noise_scale:.4f}\n')

    def fit(self, data: Data) -> Metrics:
        self.data = data

        with console.status(f'moving data to {self.device}'):
            self.data.to(self.device)
        
        start_time = time()
        self.init_privacy_mechanisms()

        if self.dp_level == 'node' and self.max_degree > 0:
            with console.status('bounding the number of neighbors per node'):
                self.data = NeighborSampler(self.max_degree)(self.data)
        else:
            with console.status('perturbing graph structure'):
                self.data.adj_t = self.graph_mechanism(self.data.adj_t, chunk_size=500)

        logging.info('training classifier...')
        metrics = self.train_classifier()

        end_time = time()
        metrics['time'] = end_time - start_time

        return metrics


    def train_classifier(self) -> Metrics:
        self.classifier.to(self.device)

        trainer = Trainer(
            epochs=self.epochs, 
            use_amp=self.use_amp, 
            val_interval=not(self.dp_level == 'node' and self.epsilon < np.inf),
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

        return metrics

    def data_loader(self, stage: TrainerStage) -> NeighborLoader:
        if self.batch_size <= 0:
            self.data.batch_size = self.data.num_nodes
            return [self.data]
        else:
            return NeighborLoader(
                data=self.data, 
                num_neighbors=[self.max_degree]*self.mp_layers, 
                input_nodes=self.data[f'{stage}_mask'],
                replace=False,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=6,
            )

    def configure_optimizers(self, model: Module) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
