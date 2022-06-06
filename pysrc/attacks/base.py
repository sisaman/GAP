import logging
from abc import ABC, abstractmethod
from typing import Annotated
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from pysrc.console import console
from pysrc.classifiers.base import Metrics
from pysrc.methods.base import MethodBase
from pysrc.methods.mlp import MLP


class AttackBase(MLP, ABC):
    def __init__(self, 
                 method: MethodBase,
                 target_train_ratio: Annotated[float, dict(help='ratio of training nodes in the target dataset')] = 0.3,
                 target_val_ratio: Annotated[float, dict(help='ratio of validation nodes in the target dataset')] = 0.1,
                 shadow_train_ratio: Annotated[float, dict(help='ratio of training nodes in the shadow dataset')] = 0.3,
                 shadow_val_ratio: Annotated[float, dict(help='ratio of validation nodes in the shadow dataset')] = 0.1,
                ):

        super().__init__(
            num_classes=2,  # either member or non-member
            hidden_dim=16,
            num_layers=3,
            activation='selu',
            batch_norm=True,
            device=method.device
        )

        self.method = method
        self.target_train_ratio = target_train_ratio
        self.target_val_ratio = target_val_ratio
        self.shadow_train_ratio = shadow_train_ratio
        self.shadow_val_ratio = shadow_val_ratio

    def reset_parameters(self):
        super().reset_parameters()
        self.method.reset_parameters()

    def execute(self, data: Data) -> Metrics:
        # split data into target and shadow dataset
        data_target, data_shadow = self.target_shadow_split(data)
        
        # train target model and obtain logits
        logging.info('step 1: training target model')
        metrics = self.method.fit(data_target)
        data_target.logits = self.method.predict()

        # train shadow model and obtain logits
        logging.info('step 2: training shadow model')
        self.method.reset_parameters()
        self.method.fit(data_shadow)
        data_shadow.logits = self.method.predict()

        # construct attack dataset
        with console.status('constructing attack dataset'):
            data_attack = self.prepare_attack_dataset(data_target, data_shadow)

        # train attack model and get attack accuracy
        logging.info('step 3: training attack model')
        attack_metrics = self.fit(data_attack)
        metrics['attack/acc'] = attack_metrics['test/acc']
        
        return metrics

    def target_shadow_split(self, data: Data) -> tuple[Data, Data]:
        data_target = Data(**data.to_dict())
        data_shadow = Data(**data.to_dict())
        
        data_target = RandomNodeSplit(
            split='test_rest',
            num_train_per_class=int(self.target_train_ratio * data.num_nodes / self.method.num_classes),
            num_val=self.target_val_ratio
        )(data_target)

        data_shadow = RandomNodeSplit(
            split='test_rest',
            num_train_per_class=int(self.shadow_train_ratio * data.num_nodes / self.method.num_classes),
            num_val=self.shadow_val_ratio
        )(data_shadow)

        logging.debug(f'target dataset: {data_target.train_mask.sum()} train nodes')
        logging.debug(f'shadow dataset: {data_shadow.train_mask.sum()} train nodes')

        return data_target, data_shadow

    def subgraph(self, data: Data, mask: Tensor) -> Data:
        return Data(
            x=data.x[mask], 
            y=data.y[mask], 
            train_mask=data.train_mask[mask], 
            val_mask=data.val_mask[mask], 
            test_mask=data.test_mask[mask],
            adj_t=data.adj_t[mask, mask],
        )
    
    def prepare_attack_dataset(self, data_target: Data, data_shadow: Data) -> Data:
        # get train+val data from shadow data
        logging.debug('preparing attack dataset: train')
        x_train_val, y_train_val = self.generate_attack_xy(data_shadow)

        # shuffle train+val data
        perm = torch.randperm(x_train_val.size(0), device=self.device)
        x_train_val, y_train_val = x_train_val[perm], y_train_val[perm]
        
        # get test data from target data
        logging.debug('preparing attack dataset: test')
        x_test, y_test = self.generate_attack_xy(data_target)
        
        # combine train+val and test data
        x = torch.cat([x_train_val, x_test], dim=0)
        y = torch.cat([y_train_val, y_test], dim=0)
        
        # create train/val/test masks
        num_total = x.size(0)
        num_train_val = x_train_val.size(0)
        num_train = int(0.8 * num_train_val)
        
        train_mask = x.new_zeros(num_total, dtype=torch.bool)
        train_mask[:num_train] = True

        val_mask = x.new_zeros(num_total, dtype=torch.bool)
        val_mask[num_train:num_train_val] = True

        test_mask = x.new_zeros(num_total, dtype=torch.bool)
        test_mask[num_train_val:] = True
        
        # create data object
        data_attack = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return data_attack

    @abstractmethod
    def generate_attack_xy(self, data: Data) -> tuple[Tensor, Tensor]: pass