import os
import ssl
from functools import partial
import torch
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, subgraph
from torch_geometric.datasets import FacebookPagePage, LastFMAsia, Coauthor, Amazon, WikiCS, Reddit2
from torch_geometric.transforms import RandomNodeSplit, Compose
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
from scipy.io import loadmat
from torch_geometric.data import Data, InMemoryDataset, download_url
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import from_scipy_sparse_matrix
from args import support_args


def load_ogb(name, transform=None, **kwargs):
    dataset = PygNodePropPredDataset(name=name, **kwargs)
    data = dataset[0]
    data.y = data.y.flatten()
    split_idx = dataset.get_idx_split()
    for split, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=bool).scatter(0, idx, True)
        split = 'val' if split == 'valid' else split
        data[f'{split}_mask'] = mask

    return [transform(data)]


class FilterTopClass:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, data):
        y = torch.nn.functional.one_hot(data.y)
        c = y.sum(dim=0).sort(descending=True)
        y = y[:, c.indices[:self.num_classes]]
        idx = y.sum(dim=1).bool()

        data.x = data.x[idx]
        data.y = y[idx].argmax(dim=1)
        data.num_nodes = data.y.size(0)
        data.edge_index, data.edge_attr = subgraph(idx, data.edge_index, data.edge_attr, relabel_nodes=True)

        if 'train_mask' in data:
            data.train_mask = data.train_mask[idx]
            data.val_mask = data.val_mask[idx]
            data.test_mask = data.test_mask[idx]

        return data


class Facebook100(InMemoryDataset):
    url = 'https://github.com/sisaman/pyg-datasets/raw/main/datasets/facebook100/'
    targets = ['status', 'gender', 'major', 'minor', 'housing', 'year']
    available_datasets = [
        "Villanova62", "UCLA26", "Tennessee95", "NYU9", "Carnegie49", "GWU54", "USF51", "Vanderbilt48", 
        "USC35", "Rutgers89", "UConn91", "MIT8", "USFCA72", "UChicago30", "UIllinois20", "UC61", "Cal65", 
        "Yale4", "Northeastern19", "Dartmouth6", "Vermont70", "Northwestern25", "William77", "Harvard1", 
        "Princeton12", "UC64", "Middlebury45", "Haverford76", "Bingham82", "UNC28", "Berkeley13", "Rochester38", 
        "Swarthmore42", "Virginia63", "WashU32", "Columbia2", "NotreDame57", "Bucknell39", "UVA16", "Maine59", 
        "MU78", "Simmons81", "MSU24", "Colgate88", "Temple83", "Cornell5", "Indiana69", "Oklahoma97", "Michigan23", 
        "BU10", "Brown11", "Auburn71", "FSU53", "UGA50", "UCF52", "Howard90", "UCSD34", "Vassar85", "Tufts18", 
        "UPenn7", "Baylor93", "UMass92", "Bowdoin47", "Maryland58", "Penn94", "Wesleyan43", "UC33", 
        "Rice31", "UCSC68", "Smith60", "Caltech36", "Hamilton46", "Oberlin44", "American75", "Mich67", 
        "Mississippi66", "Williams40", "UCSB37", "Amherst41", "Duke14", "Pepperdine86", "Wake73", "Lehigh96", 
        "Reed98", "Tulane29", "Texas84", "Wellesley22", "JMU79", "Santa74", "Wisconsin87", "Stanford3", 
        "Texas80", "UF21", "JohnsHopkins55", "Syracuse56", "BC17", "Georgetown15", "Trinity100", "Brandeis99", "Emory27"
    ]

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        self.target = 'status'
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return self.name + '.mat'

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)
        ssl._create_default_https_context = context

    def process(self):

        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names))
        features = pd.DataFrame(
            mat['local_info'][:, :-1], columns=self.targets)
        y = torch.from_numpy(LabelEncoder().fit_transform(features[self.target]))
        if 0 in features[self.target].values:
            y = y - 1

        x = features.drop(columns=self.target).replace({0: pd.NA})
        x = torch.tensor(pd.get_dummies(x).values, dtype=torch.float)
        edge_index = from_scipy_sparse_matrix(mat['A'])[0]

        # removed unlabled nodes
        subset = y >= 0
        edge_index, _ = subgraph(subset, edge_index, relabel_nodes=True, num_nodes=len(y))
        x = x[subset]
        y = y[subset]

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(y))

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Facebook100-{self.name}()'


@support_args
class Dataset:
    supported_datasets = {
        'facebook': partial(FacebookPagePage, transform=RandomNodeSplit(split='train_rest')),
        'lastfm': partial(LastFMAsia, transform=RandomNodeSplit(split='train_rest')),
        'co-ph': partial(Coauthor, name='physics', transform=RandomNodeSplit(split='train_rest')),
        'amz-comp': partial(Amazon, name='computers', transform=RandomNodeSplit(split='train_rest')),
        'wiki': partial(WikiCS, transform=RandomNodeSplit(split='train_rest')),
        'reddit': partial(Reddit2, transform=Compose([FilterTopClass(5), RandomNodeSplit(split='train_rest')])),
        'fb-harvard': partial(Facebook100, name='Harvard1', transform=RandomNodeSplit(split='train_rest')),
        # 'pubmed': partial(CitationFull, name='pubmed', transform=RandomNodeSplit(split='train_rest')),
        # 'arxiv': partial(load_ogb, name='ogbn-arxiv'),
        # 'cora': partial(CitationFull, name='cora', transform=RandomNodeSplit(split='train_rest')),
        # 'citeseer': partial(CitationFull, name='citeseer', transform=RandomNodeSplit(split='train_rest')),
        # 'github': partial(GitHub, transform=RandomNodeSplit(split='train_rest')),
        # 'co-cs': partial(Coauthor, name='cs', transform=RandomNodeSplit(split='train_rest')),
        # 'amz-photo': partial(Amazon, name='photo', transform=RandomNodeSplit(split='train_rest')),
        # 'deezer': partial(DeezerEurope, transform=RandomNodeSplit(split='train_rest')),
        # 'twitch-de': partial(Twitch, name='DE', transform=RandomNodeSplit(split='train_rest')),
    }

    def __init__(self,
                 dataset:    dict(help='name of the dataset', choices=supported_datasets) = 'reddit',
                 data_dir:   dict(help='directory to store the dataset') = './datasets',
                 feature:    dict(help='type of node feature ("raw" for original features, "rand" for random features)') = 'raw',
                 normalize:  dict(help='if set to true, row-normalizes features') = True
                 ):
        self.name = dataset
        self.data_dir = data_dir
        self.feature = feature
        self.normalize = normalize

    def load(self):
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data.edge_index, _ = remove_self_loops(data.edge_index)

        if self.feature == 'rand':
            data.x = torch.randn_like(data.x)
        elif self.feature == 'one':
            data.x = torch.ones_like(data.x)
        elif self.feature == 'pca' and data.num_features > 128:
            _, _, V = torch.pca_lowrank(data.x, q=128)
            data.x = torch.matmul(data.x, V[:, :128])

        if self.normalize:
            data.x = F.normalize(data.x, p=2., dim=-1)

        return data
