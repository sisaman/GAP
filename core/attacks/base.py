from abc import ABC, abstractmethod
from typing import Annotated
import torch
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics.functional import auroc, average_precision
from core.args.utils import remove_prefix
from core.console import console
from core.classifiers.base import Metrics
from core.methods.base import NodeClassificationBase
from core.methods.mlp import MLP


class AttackBase(MLP, ABC):
    def __init__(self, 
                 method:    NodeClassificationBase,
                 **kwargs:  Annotated[dict, dict(help='attack method kwargs', bases=[MLP], prefixes=['attack_'], exclude=['device', 'use_amp'])],
                ):

        super().__init__(
            num_classes=2,  # either member or non-member
            **remove_prefix(kwargs, prefix='attack_'),
        )

        self.method = method

    def reset_parameters(self):
        super().reset_parameters()
        self.method.reset_parameters()

    def execute(self, data: Data) -> Metrics:
        # split data into target and shadow dataset
        target_data, shadow_data = self.target_shadow_split(data)

        # train target model and obtain confidence scores
        console.info('step 1: training target model')
        target_metrics = self.method.fit(Data(**target_data.to_dict()), prefix='target/')
        target_scores = self.method.predict()
        target_data, target_scores = target_data.to('cpu'), target_scores.to('cpu')

        # train shadow model and obtain confidence scores
        console.info('step 2: training shadow model')
        self.method.reset_parameters()
        shadow_metrics = self.method.fit(Data(**shadow_data.to_dict()), prefix='shadow/')
        shadow_scores = self.method.predict()
        shadow_data, shadow_scores = shadow_data.to('cpu'), shadow_scores.to('cpu')

        # construct attack dataset
        with console.status('constructing attack dataset'):
            attack_data = self.prepare_attack_dataset(
                target_data=target_data, 
                target_scores=target_scores, 
                shadow_data=shadow_data, 
                shadow_scores=shadow_scores,
            )
            console.debug(f'attack dataset: {attack_data.train_mask.sum()} train nodes, {attack_data.val_mask.sum()} val nodes, {attack_data.test_mask.sum()} test nodes')

        # train attack model and get attack accuracy
        console.info('step 3: training attack model')
        attack_metrics = self.fit(attack_data, prefix='attack/')
        
        # compute extra attack metrics
        preds = self.predict()[attack_data.test_mask, 1]
        target = attack_data.y[attack_data.test_mask]
        attack_metrics['attack/test/auc'] = auroc(preds=preds, target=target).item() * 100
        attack_metrics['attack/test/avg_precision'] = average_precision(preds=preds, target=target).item() * 100
        attack_metrics['attack/test/advantage'] = max(0, 2 * attack_metrics['attack/test/acc'] - 100)

        # aggregate metrics
        metrics = {
            **target_metrics,
            **shadow_metrics,
            **attack_metrics,
        }
        
        return metrics

    def prepare_attack_dataset(self, target_data: Data, target_scores: Tensor, shadow_data: Data, shadow_scores: Tensor) -> Data:
        # get train+val data from shadow data
        console.debug('preparing attack dataset: train')
        x_train_val, y_train_val = self.generate_attack_samples(data=shadow_data, scores=shadow_scores)
        
        # get test data from target data
        console.debug('preparing attack dataset: test')
        x_test, y_test = self.generate_attack_samples(data=target_data, scores=target_scores)
        
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
        attack_data = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return attack_data

    @abstractmethod
    def target_shadow_split(self, data: Data) -> tuple[Data, Data]: pass

    @abstractmethod
    def generate_attack_samples(self, data: Data, scores: Tensor) -> tuple[Tensor, Tensor]: pass
