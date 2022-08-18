from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class RemoveIsolatedNodes(BaseTransform):
    def __call__(self, data: Data) -> Data:
        mask = data.y.new_zeros(data.num_nodes, dtype=bool)
        mask[data.edge_index[0]] = True
        mask[data.edge_index[1]] = True
        data = data.subgraph(mask)
        return data
