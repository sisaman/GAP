import torch
from torch_geometric.utils import remove_isolated_nodes, subgraph
from args import support_args

@support_args
class RandomSubGraphSampler(torch.utils.data.DataLoader):

    def __init__(self, data, 
                 sampling_prob: dict(help='data loader sampling probability') = 1.0, 
                 max_epochs:    dict(help='maximum number of training epochs') = 100,
                 num_workers:   dict(help='how many subprocesses to use for data loading') = 0,
                 pin_memory = False,
                 edge_sampler = False,
                 ):

        self.data = data
        self.sampling_prob = float(sampling_prob)
        self.edge_sampler = edge_sampler
        self.num_steps = max_epochs
        self.N = data.num_nodes
        self.E = data.num_edges
        self.sampler_fn = self.sample_edges if edge_sampler else self.sample_nodes

        super().__init__(self, batch_size=1, collate_fn=self.__collate__, num_workers=num_workers, pin_memory=pin_memory)

    def sample_nodes(self):
        device = self.data.x.device
        node_mask = torch.bernoulli(torch.full((self.N, ), self.sampling_prob, device=device)).bool()
        edge_index, _ = subgraph(node_mask, self.data.edge_index, relabel_nodes=True, num_nodes=self.N)
        return node_mask, edge_index

    def sample_edges(self):
        device = self.data.x.device
        edge_mask = torch.bernoulli(torch.full((self.E, ), self.sampling_prob, device=device)).bool()
        edge_index = self.data.edge_index[:, edge_mask]
        edge_index, _, node_mask = remove_isolated_nodes(edge_index, num_nodes=self.N)
        return node_mask, edge_index

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_steps

    def __collate__(self, _):
        node_mask, edge_index = self.sampler_fn()

        data = self.data.__class__()
        data.edge_index = edge_index

        for key, item in self.data:
            if key in ['num_nodes', 'edge_index']:
                continue
            if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                data[key] = item[node_mask]
            else:
                data[key] = item

        return data
