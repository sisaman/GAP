import logging
from abc import ABC, abstractmethod
from typing import Annotated
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from pysrc.args.utils import remove_prefix
from pysrc.attacks.utils import membership_advantage
from pysrc.console import console
from pysrc.classifiers.base import Metrics
from pysrc.methods.base import MethodBase
from pysrc.methods.mlp import MLP


class AttackBase(MLP, ABC):
    def __init__(self, 
                 method: MethodBase,
                 num_train_nodes: Annotated[int,  dict(help='number of training nodes in both target and shadow datasets')] = 10000,
                 num_val_nodes:   Annotated[int,  dict(help='number of validation nodes in both target and shadow datasets')] = 100,
                 **kwargs:        Annotated[dict, dict(help='extra options passed to base class', bases=[MLP], prefixes=['attack_'])]
                ):

        super().__init__(
            num_classes=2,  # either member or non-member
            **remove_prefix(kwargs, prefix='attack_'),
        )

        self.method = method
        self.num_train_nodes = num_train_nodes
        self.num_val_nodes = num_val_nodes

    def reset_parameters(self):
        super().reset_parameters()
        self.method.reset_parameters()

    def execute(self, data: Data) -> Metrics:
        # split data into target and shadow dataset
        data_target, data_shadow = self.target_shadow_split(data)
        metrics = {}
        # train target model and obtain logits
        logging.info('step 1: training target model')
        target_metrics = self.method.fit(data_target)
        data_target.logits = self.method.predict()
        data_target = data_target.to('cpu')

        # train shadow model and obtain logits
        logging.info('step 2: training shadow model')
        self.method.reset_parameters()
        shadow_metrics = self.method.fit(data_shadow)
        data_shadow.logits = self.method.predict()
        data_shadow = data_shadow.to('cpu')

        # construct attack dataset
        with console.status('constructing attack dataset'):
            data_attack = self.prepare_attack_dataset(data_target, data_shadow)
            data_attack = data_attack.to(self.device)
            logging

        # train attack model and get attack accuracy
        logging.info('step 3: training attack model')
        attack_metrics = self.fit(data_attack)

        # aggregate metrics
        metrics.update({f'target/{k}': v for k, v in target_metrics.items()})
        metrics.update({f'shadow/{k}': v for k, v in shadow_metrics.items()})
        metrics.update({f'attack/{k}': v for k, v in attack_metrics.items()})
        

        metrics['attack/adv'] = 2 * metrics['attack/test/acc'] - 100
        
        return metrics

    def target_shadow_split(self, data: Data) -> tuple[Data, Data]:
        data_target = Data(**data.to_dict())
        data_shadow = Data(**data.to_dict())
        
        data_target = RandomNodeSplit(
            split='test_rest',
            num_train_per_class=self.num_train_nodes // self.method.num_classes,
            num_val=self.num_val_nodes
        )(data_target)

        data_shadow = RandomNodeSplit(
            split='test_rest',
            num_train_per_class=self.num_train_nodes // self.method.num_classes,
            num_val=self.num_val_nodes
        )(data_shadow)

        logging.debug(f'target dataset: {data_target.train_mask.sum()} train nodes, {data_target.val_mask.sum()} val nodes')
        logging.debug(f'shadow dataset: {data_shadow.train_mask.sum()} train nodes, {data_shadow.val_mask.sum()} val nodes')

        return data_target, data_shadow

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