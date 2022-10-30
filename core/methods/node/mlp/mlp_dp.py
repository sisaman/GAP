import numpy as np
from typing import Annotated, Literal, Union
from torch.optim import Optimizer
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.methods.node import MLP
from core.privacy.algorithms.noisy_sgd import NoisySGD
from core.modules.base import Metrics, Stage
from core.data.loader import NodeDataLoader


class PrivMLP (MLP):
    """node-private MLP method"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float], 
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_grad_norm: Annotated[float, ArgInfo(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   ArgInfo(help='batch size')] = 256,
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[MLP], exclude=['batch_norm'])]
                 ):

        super().__init__(num_classes, 
            batch_norm=False, 
            batch_size=batch_size, 
            **kwargs
        )

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.num_train_nodes = None         # will be used to auto set delta

    def calibrate(self):
        self.noisy_sgd = NoisySGD(
            noise_scale=0.0, 
            dataset_size=self.num_train_nodes,
            batch_size=self.batch_size, 
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
        )

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_train_nodes)))
                console.info('delta = %.0e', delta)
            
            self.noise_scale = self.noisy_sgd.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

        self._classifier = self.noisy_sgd.prepare_module(self._classifier)

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        dataloader = super().data_loader(data, stage)
        if stage == 'train':
            dataloader.poisson_sampling = True
        return dataloader

    def configure_optimizer(self) -> Optimizer:
        optimizer = super().configure_optimizer()
        optimizer = self.noisy_sgd.prepare_optimizer(optimizer)
        return optimizer
