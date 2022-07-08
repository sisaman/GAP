from typing import Annotated, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from core.attacks.base import AttackBase
from core.console import console


class LinkStealingAttack (AttackBase):
    """link inference attack"""

    def __init__(self, 
                 max_samples: Annotated[Optional[int], dict(help='max number of samples to generate')] = None,
                 **kwargs: Annotated[dict,  dict(help='extra options passed to base class', bases=[AttackBase])]
                 ):
        super().__init__(**kwargs)
        self.max_samples = max_samples
        
    def generate_attack_samples(self, data: Data, logits: Tensor) -> tuple[Tensor, Tensor]:
        assert hasattr(data, 'logits'), 'data must have attribute "logits"'

        # convert adj_t to edge_index
        edge_index = torch.cat(data.adj_t.t().coo()[:-1]).view(2,-1)

        # exclude self loops from edge_index
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:,mask]

        # calculate sample size
        total_pos = edge_index.size(1)
        total_neg = data.num_nodes * (data.num_nodes - 1) - total_pos
        num_half = min(total_neg, total_pos)
        if self.max_samples is not None:
            num_half = min(self.max_samples // 2, num_half)

        # randomly sample num_half edges
        perm = torch.randperm(edge_index.size(1), device=self.device)[:num_half]
        pos_idx = edge_index[:, perm]
        pos_samples = torch.cat([logits[pos_idx[0]], logits[pos_idx[1]]], dim=1)

        # negative sampling
        neg_idx = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=num_half, method='sparse')
        neg_samples = torch.cat([logits[neg_idx[0]], logits[neg_idx[1]]], dim=1)
        
        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=self.device),
            torch.ones(num_half, dtype=torch.long, device=self.device),
        ])

        num_classes = x.size(1) // 2
        label_left = x[:, :num_classes].argmax(dim=1)
        label_right = x[:, num_classes:].argmax(dim=1)
        acc = label_left.eq(label_right).eq(y.bool()).float().mean()
        console.debug(f'baseline accuracy: {acc}')

        return x, y
