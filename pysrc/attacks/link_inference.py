from typing import Annotated
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from pysrc.attacks.base import AttackBase


class LinkStealingAttack (AttackBase):
    """link inference attack model"""

    def __init__(self, **kwargs: Annotated[dict,  dict(help='extra options passed to base class', bases=[AttackBase])]):
        super().__init__(**kwargs)
        
    def prepare_attack_dataset(self, data: Data) -> tuple[Tensor, Tensor]:
        assert hasattr(data, 'logits'), 'data must have attribute "logits"'

        # convert adj_t to edge_index
        edge_index = torch.cat(data.adj_t.t().coo()[:-1]).view(2,-1)
        # exclude self loops from edge_index
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:,mask]

        total_pos = edge_index.size(1)
        total_neg = data.num_nodes * (data.num_nodes - 1) - total_pos
        num_half = min(self.max_samples // 2, min(total_neg, total_pos))

        # randomly sample num_half edges
        perm = torch.randperm(edge_index.size(1), device=self.device)[:num_half]
        pos_idx = edge_index[:, perm]
        pos_samples = torch.cat([data.logits[pos_idx[0]], data.logits[pos_idx[1]]], dim=1)

        # negative sampling
        neg_idx = negative_sampling(edge_index, num_nodes=data.num_nodes, num_neg_samples=num_half, method='sparse')
        neg_samples = torch.cat([data.logits[neg_idx[0]], data.logits[neg_idx[1]]], dim=1)
        
        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=self.device),
            torch.ones(num_half, dtype=torch.long, device=self.device),
        ])

        return x, y
