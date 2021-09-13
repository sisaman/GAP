import os
from functools import partial
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import Planetoid, GitHub, FacebookPagePage, LastFMAsia
from torch_geometric.transforms import RandomNodeSplit
from args import support_args


supported_datasets = {
    'cora': partial(Planetoid, name='cora', split='full'),
    'citeseer': partial(Planetoid, name='citeseer', split='full'),
    'pubmed': partial(Planetoid, name='pubmed', split='full'),
    'github': partial(GitHub, transform=RandomNodeSplit(split='train_rest')),
    'facebook': partial(FacebookPagePage, transform=RandomNodeSplit(split='train_rest')),
    'lastfm': partial(LastFMAsia, transform=RandomNodeSplit(split='train_rest')),
}

@support_args
class Dataset:
    def __init__(self,
        dataset:    dict(help='name of the dataset', choices=supported_datasets) = 'cora',
        data_dir:   dict(help='directory to store the dataset') = './datasets',
    ):
        self.name = dataset
        self.data_dir = data_dir

    def load(self):
        data = supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.x = F.normalize(data.x, p=2., dim=-1)
        return data
