import os
from functools import partial

import dgl
from dgl.data import CitationGraphDataset
import torch.nn.functional as F



supported_datasets = {
    'cora': partial(CitationGraphDataset, name='cora', verbose=False),
    'citeseer': partial(CitationGraphDataset, name='citeseer', verbose=False),
    'pubmed': partial(CitationGraphDataset, name='pubmed', verbose=False),
}


def load_dataset(
        dataset:        dict(help='name of the dataset', option='-d', choices=supported_datasets) = 'cora',
        data_dir:       dict(help='directory to store the dataset') = './datasets',
        normalize:     dict(help='whether to unit-normalize the input node features') = True,
        ):
    g = supported_datasets[dataset](raw_dir=os.path.join(data_dir, dataset))[0]
    g = dgl.add_self_loop(dgl.remove_self_loop(g))
    g.name = dataset
    g.num_features = g.ndata['feat'].size(1)
    g.num_classes = int(g.ndata['label'].max().item()) + 1

    if normalize:
        g.ndata['feat'] = F.normalize(input=g.ndata['feat'], p=2, dim=1)
    return g
