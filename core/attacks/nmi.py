from typing import Annotated
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from core.attacks.base import AttackBase
from core.console import console
from core.data.transforms import RandomDataSplit


class NodeMembershipInference (AttackBase):
    """node membership inference attack"""

    def __init__(self, 
                 num_nodes_per_class:   Annotated[int,  dict(help='number of nodes per class in both target and shadow datasets')] = 1000,
                 retain_target_data:    Annotated[bool, dict(help='whether to retain target dataset')] = False,
                 **kwargs:              Annotated[dict,  dict(help='extra options passed to base class', bases=[AttackBase])]
                 ):

        super().__init__(**kwargs)
        self.num_nodes_per_class = num_nodes_per_class
        self.retain_target_data = retain_target_data

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
