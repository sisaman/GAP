from typing import Literal, Union
import torch
from collections.abc import Iterator
from torch_geometric.data import Data
from core.classifiers.base import Stage


class NodeDataLoader:
    def __init__(self, 
                 data: Data, 
                 stage: Stage,
                 batch_size: Union[int, Literal['full']] = 'full', 
                 shuffle: bool = True, 
                 drop_last: bool = False, 
                 poisson_sampling: bool = False):

        self.data = data
        self.stage = stage
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.poisson_sampling = poisson_sampling
        self.device = data.x.device

        if batch_size != 'full':
            self.node_indices = data[f'{stage}_mask'].nonzero().view(-1)
            self.num_nodes = self.node_indices.size(0)

    def __iter__(self) -> Iterator[Data]:
        if self.batch_size == 'full':
            yield self.data
            return

        if self.shuffle and not self.poisson_sampling:
            perm = torch.randperm(self.num_nodes, device=self.device)
            self.node_indices = self.node_indices[perm]

        for i in range(0, self.num_nodes, self.batch_size):
            if self.drop_last and i + self.batch_size > self.num_nodes:
                break

            if self.poisson_sampling:
                sampling_prob = self.batch_size / self.num_nodes
                sample_mask = torch.rand(self.num_nodes, device=self.device) < sampling_prob
                batch_nodes = self.node_indices[sample_mask]
            else:    
                batch_nodes = self.node_indices[i:i + self.batch_size]

            batch_mask = torch.zeros(self.data.num_nodes, device=self.device, dtype=torch.bool)
            batch_mask[batch_nodes] = True

            data = Data(**self.data.to_dict())
            data[f'{self.stage}_mask'] = data[f'{self.stage}_mask'] & batch_mask
            
            yield data
            
    def __len__(self) -> int:
        if self.batch_size == 'full':
            return 1
        elif self.drop_last:
            return self.num_nodes // self.batch_size
        else:
            return (self.num_nodes + self.batch_size - 1) // self.batch_size
