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
                 **kwargs:              Annotated[dict,  dict(help='extra options passed to base class', bases=[ModelBasedAttack])]
                 ):

        super().__init__(**kwargs)
        self.num_nodes_per_class = num_nodes_per_class

    def execute(self, method: NodeClassification, data: Data) -> Metrics:
        attack_metrics = super().execute(method, data)
        return {
            **self.shadow_metrics,
            **attack_metrics,
        }

    def prepare_attack_dataset(self, method: NodeClassification, data: Data) -> Data:
        shadow_data = RandomDataSplit(
            num_nodes_per_class=self.num_nodes_per_class,
            train_ratio=0.4,
            test_ratio=0.4
        )(data)

        # train shadow model and obtain confidence scores
        console.info('training shadow model')
        method.reset_parameters()
        self.shadow_metrics = method.fit(Data(**shadow_data.to_dict()), prefix='shadow/')
        shadow_scores = method.predict()
        shadow_data, shadow_scores = shadow_data.to('cpu'), shadow_scores.to('cpu')
        
        # get train+val data from shadow data
        console.debug('preparing attack dataset')
        x, y = self.generate_attack_samples(data=shadow_data, scores=shadow_scores)
        
        # train test split
        num_total = x.size(0)
        num_train = int(num_total * 0.6)
        num_test = int(num_total * 0.3)
        
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
