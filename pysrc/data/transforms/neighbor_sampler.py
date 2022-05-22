import torch
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor
import torch.utils.cpp_extension


class NeighborSampler(BaseTransform):
    def __init__(self, max_degree: int):
        self.max_deg = max_degree
        
        try:
            edge_sampler = torch.ops.my_ops.sample_edge
        except RuntimeError:
            torch.utils.cpp_extension.load(
                name="sampler",
                sources=['csrc/sampler.cpp'],
                build_directory='csrc',
                is_python_module=False,
                verbose=False,
            )
            edge_sampler = torch.ops.my_ops.sample_edge
        
        self.edge_sampler = edge_sampler

    def __call__(self, data):
        N = data.num_nodes
        E = data.num_edges
        adj = data.adj_t.t()
        device = adj.device()
        row, col, _ = adj.coo()
        perm = torch.randperm(E)
        row, col = row[perm], col[perm]
        row, col = self.edge_sampler(row.tolist(), col.tolist(), N, self.max_deg)
        adj = SparseTensor(row=row, col=col).to(device)
        data.adj_t = adj.t()
        return data
