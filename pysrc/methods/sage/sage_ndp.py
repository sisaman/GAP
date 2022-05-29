import logging
import numpy as np
from typing import Annotated, Union, Literal
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from pysrc.console import console
from pysrc.data.transforms import NeighborSampler
from pysrc.methods.sage.sage_inf import SAGEINF
from pysrc.classifiers.base import Metrics, Stage
from pysrc.privacy.algorithms import GNNBasedNoisySGD
from pysrc.privacy.mechanisms import GaussianMechanism
from pysrc.privacy.mechanisms import ComposedNoisyMechanism


class SAGENDP (SAGEINF):
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, dict(help='DP epsilon parameter', option='-e')] = np.inf,
                 delta:         Annotated[Union[Literal['auto'], float], dict(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 max_degree:    Annotated[int,   dict(help='max degree to sample per each node')] = 100,
                 max_grad_norm: Annotated[float, dict(help='maximum norm of the per-sample gradients')] = 1.0,
                 batch_size:    Annotated[int,   dict(help='batch size')] = 256,
                 **kwargs:      Annotated[dict, dict(help='extra options passed to base class', bases=[SAGEINF], exclude=['batch_norm', 'mp_layers'])]
                 ):

        super().__init__(num_classes, mp_layers=1, batch_size=batch_size, batch_norm=False, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.max_degree = max_degree
        self.max_grad_norm = max_grad_norm
        
        self.num_train_nodes = None         # will be used to auto set delta
        self.classifier.normalize = True    # required to bound sensitivity
        self.trainer.val_interval = 0       # disable validation

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
                logging.info('delta = %.0e', delta)
            
            self.noise_scale = composed_mechanism.calibrate(eps=self.epsilon, delta=delta)
            logging.info(f'noise scale: {self.noise_scale:.4f}\n')

        self.classifier = self.noisy_sgd.prepare_module(self.classifier)

    def fit(self, data: Data) -> Metrics:
        num_train_nodes = data.train_mask.sum().item()

        if num_train_nodes != self.num_train_nodes:
            self.num_train_nodes = num_train_nodes
            self.calibrate()

        with console.status('bounding the number of neighbors per node'):
            data = NeighborSampler(self.max_degree)(data)

        return super().fit(data)

    def data_loader(self, stage: Stage) -> NeighborLoader:
        dataloader = super().data_loader(stage)
        if stage == 'train':
            dataloader = self.noisy_sgd.prepare_dataloader(dataloader)
        return dataloader

    def configure_optimizer(self) -> Optimizer:
        optimizer = super().configure_optimizer()
        optimizer = self.noisy_sgd.prepare_optimizer(optimizer)
        return optimizer
