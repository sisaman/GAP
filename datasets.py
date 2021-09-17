import os
from functools import partial
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops
from torch_geometric.datasets import CitationFull, GitHub, FacebookPagePage, LastFMAsia, Coauthor, Amazon, WikiCS, DeezerEurope, Twitch
from torch_geometric.transforms import RandomNodeSplit
from args import support_args


@support_args
class Dataset:
    supported_datasets = {
        'cora': partial(CitationFull, name='cora', transform=RandomNodeSplit(split='train_rest')),
        'citeseer': partial(CitationFull, name='citeseer', transform=RandomNodeSplit(split='train_rest')),
        'pubmed': partial(CitationFull, name='pubmed', transform=RandomNodeSplit(split='train_rest')),
        'github': partial(GitHub, transform=RandomNodeSplit(split='train_rest')),
        'facebook': partial(FacebookPagePage, transform=RandomNodeSplit(split='train_rest')),
        'lastfm': partial(LastFMAsia, transform=RandomNodeSplit(split='train_rest')),
        'co-cs': partial(Coauthor, name='cs', transform=RandomNodeSplit(split='train_rest')),
        'co-ph': partial(Coauthor, name='physics', transform=RandomNodeSplit(split='train_rest')),
        'amz-comp': partial(Amazon, name='computers', transform=RandomNodeSplit(split='train_rest')),
        'amz-photo': partial(Amazon, name='photo', transform=RandomNodeSplit(split='train_rest')),
        'wiki': partial(WikiCS, transform=RandomNodeSplit(split='train_rest')),
        'deezer': partial(DeezerEurope, transform=RandomNodeSplit(split='train_rest')),
        'twitch-de': partial(Twitch, name='DE', transform=RandomNodeSplit(split='train_rest')),
    }

    def __init__(self,
        dataset:    dict(help='name of the dataset', choices=supported_datasets) = 'cora',
        data_dir:   dict(help='directory to store the dataset') = './datasets',
    ):
        self.name = dataset
        self.data_dir = data_dir

    def load(self):
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.x = F.normalize(data.x, p=2., dim=-1)

        return data
