import torch
from torch_geometric.utils import remove_isolated_nodes, subgraph
from args import support_args

@support_args
class RandomSubGraphSampler(torch.utils.data.DataLoader):

    def __init__(self, data, 
                 sampling_rate: dict(help='data loader sampling probability') = 1.0, 
                 epochs:    dict(help='number of training epochs') = 100,
                 num_workers:   dict(help='how many subprocesses to use for data loading') = 0,
                 pin_memory = False,
                 use_edge_sampling = False,
                 transform=None,
                 ):

        self.sampling_rate = float(sampling_rate)
        self.use_edge_sampling = use_edge_sampling
        self.transform = transform
        self.num_steps = epochs
        self.N = data.num_nodes
        self.E = data.num_edges
        pin_memory = pin_memory and (sampling_rate < 1)
        self.sampler_fn = self.sample_edges if use_edge_sampling else self.sample_nodes
        self.data = transform(data) if transform and sampling_rate == 1.0 else data

        super().__init__(self, batch_size=1, collate_fn=self.__collate__, num_workers=num_workers, pin_memory=pin_memory)

    def sample_nodes(self):
        device = self.data.x.device
        node_mask = torch.bernoulli(torch.full((self.N, ), self.sampling_rate, device=device)).bool()
        edge_index, _ = subgraph(node_mask, self.data.edge_index, relabel_nodes=True, num_nodes=self.N)
        return node_mask, edge_index

    def sample_edges(self):
        device = self.data.x.device
        edge_mask = torch.bernoulli(torch.full((self.E, ), self.sampling_rate, device=device)).bool()
        edge_index = self.data.edge_index[:, edge_mask]
        edge_index, _, node_mask = remove_isolated_nodes(edge_index, num_nodes=self.N)
        return node_mask, edge_index

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_steps

    def __collate__(self, _):
        data = self.data
        
        if self.sampling_rate < 1.0:
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

            if self.transform:
                data = self.transform(data)

        return data
