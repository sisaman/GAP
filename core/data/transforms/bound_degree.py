import torch
import torch.utils.cpp_extension
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.loader.utils import to_csc, filter_data


class BoundOutDegree(BaseTransform):
    def __init__(self, max_out_degree: int):
        self.num_neighbors = max_out_degree
        self.with_replacement = False

    def __call__(self, data: Data) -> Data:
        data.adj_t = data.adj_t.t()
        data = self.sample(data)
        data.adj_t = data.adj_t.t()
        return data

    def sample(self, data: Data) -> Data:
        colptr, row, perm = to_csc(data, device='cpu')
        index = torch.arange(0, data.num_nodes-1, dtype=int)
        sample_fn = torch.ops.torch_sparse.neighbor_sample
        node, row, col, edge = sample_fn(
            colptr, row, index, [self.num_neighbors], self.with_replacement, True
        )
        data = filter_data(data, node, row, col, edge, perm)
        return data



class BoundDegree(BaseTransform):
    def __init__(self, max_degree: int):
        self.max_deg = max_degree
        
        try:
            edge_sampler = torch.ops.my_ops.sample_edge
        except (AttributeError, RuntimeError):
            torch.utils.cpp_extension.load(
                name="sampler",
                sources=['csrc/sampler.cpp'],
                build_directory='csrc',
                is_python_module=False,
                verbose=False,
            )
            edge_sampler = torch.ops.my_ops.sample_edge
        
        self.edge_sampler = edge_sampler

    def __call__(self, data: Data) -> Data:
        N = data.num_nodes
        E = data.num_edges
        adj: SparseTensor = data.adj_t.t()
        device = adj.device()
        row, col, _ = adj.coo()
        perm = torch.randperm(E)
        row, col = row[perm], col[perm]
        row, col = self.edge_sampler(row.tolist(), col.tolist(), N, self.max_deg)
        adj = SparseTensor(row=row, col=col).to(device)
        data.adj_t = adj.t()
        return data
