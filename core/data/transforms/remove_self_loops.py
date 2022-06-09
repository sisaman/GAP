from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class RemoveSelfLoops(BaseTransform):
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index, _ = remove_self_loops(data.edge_index)
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.remove_diag()
        return data
