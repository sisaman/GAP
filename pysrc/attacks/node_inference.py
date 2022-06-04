from typing import Annotated
import torch
from torch import Tensor
from torch_geometric.data import Data
from pysrc.attacks.base import AttackBase


class NodeMembershipInference (AttackBase):
    """node membership inference attack model"""

    def __init__(self, **kwargs: Annotated[dict,  dict(help='extra options passed to base class', bases=[AttackBase])]):
        super().__init__(**kwargs)
        
    def prepare_attack_dataset(self, data: Data) -> tuple[Tensor, Tensor]:
        assert hasattr(data, 'logits'), 'data must have attribute "logits"'

        num_train = data.train_mask.sum()
        num_test = data.test_mask.sum()
        num_half = min(self.max_samples // 2, min(num_train, num_test))

        perm = torch.randperm(num_train, device=self.device)[:num_half]
        pos_samples = data.logits[data.train_mask][perm]

        perm = torch.randperm(num_test, device=self.device)[:num_half]
        neg_samples = data.logits[data.test_mask][perm]

        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=self.device),
            torch.ones(num_half, dtype=torch.long, device=self.device),
        ])

        return x, y
