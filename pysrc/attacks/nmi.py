import logging
from typing import Annotated, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from pysrc.attacks.base import AttackBase


class NodeMembershipInference (AttackBase):
    """node membership inference attack"""

    def __init__(self, 
                 **kwargs: Annotated[dict,  dict(help='extra options passed to base class', bases=[AttackBase])]
                 ):
        super().__init__(**kwargs)
        
    def generate_attack_xy(self, data: Data) -> tuple[Tensor, Tensor]:
        assert hasattr(data, 'logits'), 'data must have attribute "logits"'

        num_train = data.train_mask.sum()
        num_test = data.test_mask.sum()
        num_half = min(num_train, num_test)

        perm = torch.randperm(num_train, device=self.device)[:num_half]
        pos_samples = data.logits[data.train_mask][perm]

        perm = torch.randperm(num_test, device=self.device)[:num_half]
        neg_samples = data.logits[data.test_mask][perm]

        pos_entropy = torch.distributions.Categorical(probs=pos_samples).entropy().mean()
        neg_entropy = torch.distributions.Categorical(probs=neg_samples).entropy().mean()

        logging.debug(f'pos_entropy: {pos_entropy:.4f}')
        logging.debug(f'neg_entropy: {neg_entropy:.4f}')

        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=self.device),
            torch.ones(num_half, dtype=torch.long, device=self.device),
        ])

        return x, y
