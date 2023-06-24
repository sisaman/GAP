import numpy as np
from typing import Annotated, Optional, Union, Literal
import torch
from torch.optim import Optimizer
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.data.loader import NodeDataLoader
from core.data.transforms import BoundDegree
from core.methods.node import SAGE
from core.modules.base import Metrics, Stage
from core.privacy.algorithms import GNNBasedNoisySGD
from core.privacy.mechanisms import GaussianMechanism
from core.privacy.mechanisms import ComposedNoisyMechanism


class NodePrivSAGE (SAGE):
    """node-private GraphSAGE method"""
    
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   ArgInfo(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[SAGE], exclude=['batch_norm', 'mp_layers', 'val_interval'])]
                 ):

        super().__init__(
            num_classes=num_classes, 
            batch_size=batch_size, 
            batch_norm=False, 
            mp_layers=1, 
            val_interval=0,
            **kwargs
        )
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm
        
        self.num_train_nodes = None         # will be used to auto set delta
        self.classifier.normalize = True    # required to bound sensitivity

    def calibrate(self):
        self.noisy_sgd = GNNBasedNoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size, 
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            max_degree=self.max_degree,
        )

        self.noisy_aggr_gm = GaussianMechanism(noise_scale=0.0)
        composed_mechanism = ComposedNoisyMechanism(
            noise_scale=0.0,
            mechanism_list=[self.noisy_sgd, self.noisy_aggr_gm], 
            coeff_list=[1,1]
        )

        if hasattr(self, 'noisy_aggr_hook'):
            self.noisy_aggr_hook.remove()

        self.noisy_aggr_hook = self.classifier.gnn.convs[0].register_message_and_aggregate_forward_hook(
            lambda module, inputs, output: 
                self.noisy_aggr_gm(data=output, sensitivity=np.sqrt(self.max_degree)) if not module.training else output
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e' % delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self._classifier = self.noisy_sgd.prepare_module(self._classifier)

    def sample_neighbors(self, data: Data) -> Data:
        data = data.to(self.device, non_blocking=True)
        with console.status('bounding the number of neighbors per node'):
            data = BoundDegree(self.max_degree)(data)
        return data

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        data = self.sample_neighbors(data)
        return super().fit(data, prefix=prefix)

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if data is not None and data != self.data:
            data = self.sample_neighbors(data)
        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is not None and data != self.data:
            data = self.sample_neighbors(data)
        return super().predict(data)

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        dataloader = super().data_loader(data, stage)
        if stage == 'train':
            dataloader.hops = 1
            dataloader.poisson_sampling = False
        return dataloader

    def configure_optimizer(self) -> Optimizer:
        optimizer = super().configure_optimizer()
        optimizer = self.noisy_sgd.prepare_optimizer(optimizer)
        return optimizer
