import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_geometric.loader.utils import to_csc, filter_data


class NeighborSampler(BaseTransform):
    def __init__(self, max_degree: int):
        self.num_neighbors = max_degree
        self.with_replacement = False

    def __call__(self, data: Data) -> Data:
        data.adj_t = data.adj_t.t()
        data = self.sample(data)
        data.adj_t = data.adj_t.t()
        return data

    def sample(self, data: Data) -> Data:
        colptr, row, perm = to_csc(data, device='cpu')
        index = torch.range(0, data.num_nodes-1, dtype=int)
        sample_fn = torch.ops.torch_sparse.neighbor_sample
        node, row, col, edge = sample_fn(
            colptr, row, index, [self.num_neighbors], self.with_replacement, True
        )
        data = filter_data(data, node, row, col, edge, perm)
        return data
