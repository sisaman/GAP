from typing import Annotated
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import Data
from core.args.utils import ArgInfo
from core.attacks.base import ModelBasedAttack
from core.modules.base import Metrics
from core import console
from core.data.transforms import RandomDataSplit
from core.methods.node.base import NodeClassification


class NodeMembershipInference (ModelBasedAttack):
    """node membership inference attack"""

    def __init__(self, 
                 test_on_target:        Annotated[bool, ArgInfo(help='whether to test the attack model on the data from target model')] = False,
                 num_nodes_per_class:   Annotated[int,  ArgInfo(help='number of nodes per class in both target and shadow datasets')] = 1000,
                 **kwargs:              Annotated[dict, ArgInfo(help='extra options passed to base class', bases=[ModelBasedAttack])]
                 ):

        super().__init__(**kwargs)
        self.test_on_target = test_on_target
        self.num_nodes_per_class = num_nodes_per_class

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        attack_metrics = super().execute(method, data)
        attack_metrics = self.shadow_metrics | attack_metrics
        if self.test_on_target:
            attack_metrics = self.target_metrics | attack_metrics
        return attack_metrics

    def prepare_attack_dataset(self, method: NodeClassification, data: Data) -> Data:
        if self.test_on_target:
            # train target model and obtain confidence scores
            console.info('training target model')
            method.reset_parameters()
            target_data = Data(**data.to_dict())
            self.target_metrics = method.fit(Data(**target_data.to_dict()), prefix='target/')
            target_scores = method.predict()
            target_data, target_scores = target_data.to('cpu'), target_scores.to('cpu')

        # train shadow model and obtain confidence scores
        console.info('training shadow model')
        method.reset_parameters()
        shadow_data = RandomDataSplit(
            num_nodes_per_class=self.num_nodes_per_class,
            train_ratio=0.4,
            test_ratio=0.4
        )(data)
        self.shadow_metrics = method.fit(Data(**shadow_data.to_dict()), prefix='shadow/')
        shadow_scores = method.predict()
        shadow_data, shadow_scores = shadow_data.to('cpu'), shadow_scores.to('cpu')
        
        # get attack data from shadow data and scores
        console.debug('preparing attack dataset')
        x, y = self.generate_attack_samples(data=shadow_data, scores=shadow_scores)

        if self.test_on_target:
            x_test, y_test = self.generate_attack_samples(data=target_data, scores=target_scores)
            x = torch.cat([x, x_test], dim=0)
            y = torch.cat([y, y_test], dim=0)
            num_test = x_test.size(0)
            num_train = int((x.size(0) - num_test) * 0.8)
        else:
            num_train = int(x.size(0) * 0.6)
            num_test = int(x.size(0) * 0.3)
        
        # train test split
        num_total = x.size(0)
        train_mask = y.new_zeros(num_total, dtype=torch.bool)
        train_mask[:num_train] = True

        val_mask = y.new_zeros(num_total, dtype=torch.bool)
        val_mask[num_train:-num_test] = True

        test_mask = x.new_zeros(num_total, dtype=torch.bool)
        test_mask[-num_test:] = True

        # create data object
        attack_data = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        return attack_data
        
    def generate_attack_samples(self, data: Data, scores: Tensor) -> tuple[Tensor, Tensor]:
        num_classes = scores.size(-1)
        num_train = data.train_mask.sum()
        num_test = data.test_mask.sum()
        num_half = min(num_train, num_test)

        labels = F.one_hot(data.y, num_classes).float()
        samples = torch.cat([scores, labels], dim=1)
        device = samples.device

        perm = torch.randperm(num_train, device=device)[:num_half]
        pos_samples = samples[data.train_mask][perm]

        perm = torch.randperm(num_test, device=device)[:num_half]
        neg_samples = samples[data.test_mask][perm]

        pos_entropy = Categorical(probs=pos_samples[:, :num_classes]).entropy().mean()
        neg_entropy = Categorical(probs=neg_samples[:, :num_classes]).entropy().mean()

        console.debug(f'pos_entropy: {pos_entropy:.4f}, neg_entropy: {neg_entropy:.4f}')

        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=device),
            torch.ones(num_half, dtype=torch.long, device=device),
        ])

        # shuffle data
        perm = torch.randperm(2 * num_half, device=device)
        x, y = x[perm], y[perm]

        return x, y
