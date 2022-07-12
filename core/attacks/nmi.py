from typing import Annotated
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from core.attacks.base import AttackBase
from core.console import console


class NodeMembershipInference (AttackBase):
    """node membership inference attack"""

    def __init__(self, **kwargs: Annotated[dict,  dict(help='extra options passed to base class', bases=[AttackBase])]):
        super().__init__(**kwargs)
        
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
