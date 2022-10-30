import numpy as np
import torch
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from opacus.optimizers import DPOptimizer
from core import console
from core.args.utils import ArgInfo
from core.data.loader import NodeDataLoader
from core.methods.node import GAP
from core.privacy.mechanisms import ComposedNoisyMechanism
from core.privacy.algorithms import PMA, NoisySGD
from core.data.transforms import BoundOutDegree
from core.modules.base import Metrics, Stage


class NodePrivGAP (GAP):
    """node-private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[GAP], exclude=['batch_norm'])]
                 ):

        super().__init__(num_classes, 
            batch_norm=False, 
            batch_size=batch_size, 
            **kwargs
        )
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm

        self.num_train_nodes = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)

        self.encoder_noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.encoder_epochs,
            max_grad_norm=self.max_grad_norm,
        )

        self.classifier_noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes, 
            batch_size=self.batch_size, 
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
        )

        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=0.0,
            mechanism_list=[
                self.encoder_noisy_sgd, 
                self.pma_mechanism, 
                self.classifier_noisy_sgd
            ]
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self._encoder = self.encoder_noisy_sgd.prepare_module(self._encoder)
        self._classifier = self.classifier_noisy_sgd.prepare_module(self._classifier)

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('bounding the number of neighbors per node'):
            data = BoundOutDegree(self.max_degree)(data)
        return super().compute_aggregations(data)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        x = matmul(adj_t, x)
        x = self.pma_mechanism(x, sensitivity=np.sqrt(self.max_degree))
        return x

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        dataloader = super().data_loader(data, stage)
        if stage == 'train':
            dataloader.poisson_sampling = True
        return dataloader

    def configure_optimizer(self) -> DPOptimizer:
        optimizer = super().configure_optimizer()
        optimizer = self.classifier_noisy_sgd.prepare_optimizer(optimizer)
        return optimizer

    def configure_encoder_optimizer(self) -> DPOptimizer:
        optimizer = super().configure_encoder_optimizer()
        optimizer = self.encoder_noisy_sgd.prepare_optimizer(optimizer)
        return optimizer
