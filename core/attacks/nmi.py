from typing import Annotated
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from core.attacks.base import ModelBasedAttack
from core.classifiers.base import Metrics
from core.console import console
from core.data.transforms import RandomDataSplit
from core.methods.base import NodeClassification


class NodeMembershipInference (ModelBasedAttack):
    """node membership inference attack"""

    def __init__(self, 
                 num_nodes_per_class:   Annotated[int,  dict(help='number of nodes per class in both target and shadow datasets')] = 1000,
                 retain_target_data:    Annotated[bool, dict(help='whether to retain target dataset')] = False,
                 **kwargs:              Annotated[dict,  dict(help='extra options passed to base class', bases=[ModelBasedAttack])]
                 ):

        super().__init__(**kwargs)
        self.num_nodes_per_class = num_nodes_per_class
        self.retain_target_data = retain_target_data

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        attack_metrics = super().execute(method, data)
        return {
            **self.target_metrics,
            **self.shadow_metrics,
            **attack_metrics,
        }

    def prepare_attack_dataset(self, method: NodeClassification, data: Data) -> Data:
        # split data into target and shadow dataset
        target_data, shadow_data = self.target_shadow_split(data)

        # train target model and obtain confidence scores
        console.info('step 1: training target model')
        method.reset_parameters()
        self.target_metrics = method.fit(Data(**target_data.to_dict()), prefix='target/')
        target_scores = method.predict()
        target_data, target_scores = target_data.to('cpu'), target_scores.to('cpu')

        # train shadow model and obtain confidence scores
        console.info('step 2: training shadow model')
        method.reset_parameters()
        self.shadow_metrics = method.fit(Data(**shadow_data.to_dict()), prefix='shadow/')
        shadow_scores = method.predict()
        shadow_data, shadow_scores = shadow_data.to('cpu'), shadow_scores.to('cpu')
        
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

    def target_shadow_split(self, data: Data) -> tuple[Data, Data]:
        target_data = Data(**data.to_dict())
        shadow_data = Data(**data.to_dict())
        
        if not self.retain_target_data:
            target_data = RandomDataSplit(
                num_nodes_per_class=self.num_nodes_per_class,
                train_ratio=0.4,
                test_ratio=0.4
            )(target_data)

        shadow_data = RandomDataSplit(
            num_nodes_per_class=self.num_nodes_per_class,
            train_ratio=0.4,
            test_ratio=0.4
        )(shadow_data)

        console.debug(f'target dataset: {target_data.train_mask.sum()} train nodes, {target_data.val_mask.sum()} val nodes, {target_data.test_mask.sum()} test nodes')
        console.debug(f'shadow dataset: {shadow_data.train_mask.sum()} train nodes, {shadow_data.val_mask.sum()} val nodes, {shadow_data.test_mask.sum()} test nodes')

        return target_data, shadow_data
        
    def generate_attack_samples(self, data: Data, scores: Tensor) -> tuple[Tensor, Tensor]:
        num_classes = scores.size(-1)
        num_train = data.train_mask.sum()
        num_test = data.test_mask.sum()
        num_half = min(num_train, num_test)

        labels = F.one_hot(scores.argmax(dim=1), num_classes).float()
        samples = torch.cat([scores, labels], dim=1)

        perm = torch.randperm(num_train, device=self.device)[:num_half]
        pos_samples = samples[data.train_mask][perm]

        perm = torch.randperm(num_test, device=self.device)[:num_half]
        neg_samples = samples[data.test_mask][perm]

        pos_entropy = torch.distributions.Categorical(probs=pos_samples[:, :num_classes]).entropy().mean()
        neg_entropy = torch.distributions.Categorical(probs=neg_samples[:, :num_classes]).entropy().mean()

        console.debug(f'pos_entropy: {pos_entropy:.4f}, neg_entropy: {neg_entropy:.4f}')

        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=self.device),
            torch.ones(num_half, dtype=torch.long, device=self.device),
        ])

        # shuffle data
        perm = torch.randperm(2 * num_half, device=self.device)
        x, y = x[perm], y[perm]

        return x, y
