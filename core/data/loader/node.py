from typing import Literal, Optional, Union
import torch
from collections.abc import Iterator
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.transforms import ToSparseTensor
from core.modules.base import Stage


class NodeDataLoader:
    """ A fast dataloader for node-wise training.

    Args:
        data (Data): The graph data object.
        stage (Stage): Training stage. One of 'train', 'val', 'test'.
        batch_size (int or 'full', optional): The batch size.
            If set to 'full', the entire graph is used as a single batch.
            (default: 'full')
        hops (int, optional): The number of hops to sample neighbors.
            If set to None, all neighbors are included. (default: None)
        shuffle (bool, optional): If set to True, the nodes are shuffled
            before batching. (default: True)
        drop_last (bool, optional): If set to True, the last batch is
            dropped if it is smaller than the batch size. (default: False)
        poisson_sampling (bool, optional): If set to True, poisson sampling
            is used to sample nodes. (default: False)
    """
    def __init__(self, 
                 data: Data, 
                 stage: Stage,
                 batch_size: Union[int, Literal['full']] = 'full', 
                 hops: Optional[int] = None,
                 shuffle: bool = True, 
                 drop_last: bool = False, 
                 poisson_sampling: bool = False):

        self.data = data
        self.stage = stage
        self.batch_size = batch_size
        self.hops = hops
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

            if self.hops is None:
                batch_mask = torch.zeros(self.data.num_nodes, device=self.device, dtype=torch.bool)
                batch_mask[batch_nodes] = True

                data = Data(**self.data.to_dict())
                data[f'{self.stage}_mask'] = data[f'{self.stage}_mask'] & batch_mask
            else:
                if not hasattr(self, 'edge_index'):
                    self.edge_index = torch.stack(self.data.adj_t.t().coo()[:2], dim=0)

                subset, batch_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx=batch_nodes, 
                    num_hops=self.hops, 
                    edge_index=self.edge_index, 
                    relabel_nodes=True, 
                    num_nodes=self.data.num_nodes
                )

                batch_mask = torch.zeros(subset.size(0), device=self.device, dtype=torch.bool)
                batch_mask[mapping] = True

                data = Data(
                    x=self.data.x[subset],
                    y=self.data.y[subset],
                    edge_index=batch_edge_index,
                )
                data[f'{self.stage}_mask'] = batch_mask
                data = ToSparseTensor()(data)
            
            yield data
            
    def __len__(self) -> int:
        if self.batch_size == 'full':
            return 1
        elif self.drop_last:
            return self.num_nodes // self.batch_size
        else:
            return (self.num_nodes + self.batch_size - 1) // self.batch_size
