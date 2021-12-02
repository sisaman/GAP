import os
import ssl
from functools import partial
from rich.highlighter import ReprHighlighter
from rich import box
from console import console
from rich.table import Table
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, remove_self_loops, subgraph
from torch_geometric.datasets import FacebookPagePage, LastFMAsia, Reddit2
from torch_geometric.transforms import RandomNodeSplit, Compose, BaseTransform, RemoveIsolatedNodes, ToSparseTensor
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
from scipy.io import loadmat
from torch_geometric.data import Data, InMemoryDataset, download_url
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from torch_geometric.utils import from_scipy_sparse_matrix, remove_isolated_nodes
from torch_geometric.nn import knn_graph
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

    if transform:
        data = transform(data)

    return [data]


class RandomSubGraphSampler(BaseTransform):
    def __init__(self, sampling_rate=1.0, edge_sampling=True):
        self.sampling_rate = float(sampling_rate)
        self.sampler_fn = self.edge_sampler if edge_sampling else self.node_sampler

    def node_sampler(self, data: Data):
        node_mask = torch.rand(data.num_nodes) < self.sampling_rate
        edge_index, _ = subgraph(node_mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        return node_mask, edge_index

    def edge_sampler(self, data: Data):
        edge_mask = torch.rand(data.num_edges) < self.sampling_rate
        edge_index = self.data.edge_index[:, edge_mask]
        edge_index, _, node_mask = remove_isolated_nodes(edge_index, num_nodes=data.num_nodes)
        return node_mask, edge_index

    def __call__(self, data: Data):        
        if self.sampling_rate < 1.0:
            node_mask, edge_index = self.sampler_fn()
            data.edge_index = edge_index

            n = data.num_nodes
            for key, item in data:
                if torch.is_tensor(item) and item.size(0) == n:
                    print(key)
                    data[key] = item[node_mask]

        return data


class FilterClass(BaseTransform):
    def __init__(self, top_k=None, include=None):
        assert top_k is None or top_k > 0
        assert include is None or len(include) > 0
        self.top_k = top_k
        self.include = include

    def __call__(self, data):
        num_classes = data.y.max() + 1
        include = list(range(num_classes)) if self.include is None else self.include
        top_k = len(include) if self.top_k is None else self.top_k

        y = torch.nn.functional.one_hot(data.y)
        y = y[:, include]
        c = y.sum(dim=0).sort(descending=True)
        y = y[:, c.indices[:top_k]]
        idx = y.sum(dim=1).bool()

        data.x = data.x[idx]
        data.y = y[idx].argmax(dim=1)
        # data.num_nodes = data.y.size(0)
        data.edge_index, data.edge_attr = subgraph(idx, data.edge_index, data.edge_attr, relabel_nodes=True)

        if 'train_mask' in data:
            data.train_mask = data.train_mask[idx]
            data.val_mask = data.val_mask[idx]
            data.test_mask = data.test_mask[idx]

        return data


class AddKNNGraph(BaseTransform):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def __call__(self, data):
        edge_index = knn_graph(x=data.x, k=self.k, num_workers=6)
        data.edge_index = torch.cat([data.edge_index, edge_index], dim=1)
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

    def __init__(self, root, name, target='status', transform=None, pre_transform=None):
        self.name = name
        self.target = target
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
        return f'data-{self.name}-{self.target}.pt'

    def download(self):
        context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        download_url(f'{self.url}/{self.raw_file_names}', self.raw_dir)
        ssl._create_default_https_context = context

    def process(self):

        mat = loadmat(os.path.join(self.raw_dir, self.raw_file_names))
        features = pd.DataFrame(mat['local_info'][:, :-1], columns=self.targets)
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

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'Facebook100-{self.name}()'


@support_args
class Dataset:
    supported_datasets = {
        # main datasets
        'reddit': partial(Reddit2, transform=FilterClass(6)),
        'facebook': partial(Facebook100, name='UIllinois20', target='year', transform=FilterClass(5)),
        'products': partial(load_ogb, name='ogbn-products', transform=FilterClass(include=[7,  6,  3, 12,  2])),

        # backup datasets
        'fb-pages': FacebookPagePage,
        'lastfm': partial(LastFMAsia, transform=FilterClass(10)),
        # 'amz-comp': partial(Amazon, name='computers'),
        # 'amz-photo': partial(Amazon, name='photo'),
        # 'fb-penn': partial(Facebook100, name='UPenn7', target='status'),
        # 'fb-texas': partial(Facebook100, name='Texas84', target='gender'),

        # other datasets
        # 'co-ph': partial(Coauthor, name='physics'),
        # 'co-cs': partial(Coauthor, name='cs'),
        # 'wiki': WikiCS,
        # 'fb-indiana': partial(Facebook100, name='Indiana69', target='major', transform=FilterClass(10)),
        # 'fb-harvard': partial(Facebook100, name='Harvard1', target='housing', transform=FilterClass(12)),
        # 'pubmed': partial(CitationFull, name='pubmed'),
        # 'cora': partial(CitationFull, name='cora'),
        # 'citeseer': partial(CitationFull, name='citeseer'),
        # 'github': GitHub,
        # 'flickr': Flickr,
        # 'pattern': partial(load_gnn_benchmark, name='PATTERN'),
        # 'cluster': partial(load_gnn_benchmark, name='CLUSTER'),
        # 'imdb': load_imdb,
        # 'BlogCatalog': partial(AttributedGraphDataset, name='BlogCatalog'),
        # 'Flickr': partial(AttributedGraphDataset, name='Flickr'),
        # 'Facebook': partial(AttributedGraphDataset, name='Facebook'),
        # 'Twitter': partial(AttributedGraphDataset, name='Twitter'),
        # 'TWeibo': partial(AttributedGraphDataset, name='TWeibo'),
        # 'arxiv': partial(load_ogb, name='ogbn-arxiv', transform=ToUndirected()),
        # 'deezer': DeezerEurope,
        # 'twitch-de': partial(Twitch, name='DE'),
    }

    supported_features = {'original', 'randproj16', 'randproj32', 'randproj64'}

    def __init__(self,
                 dataset:    dict(help='name of the dataset', choices=supported_datasets) = 'facebook',
                 data_dir:   dict(help='directory to store the dataset') = './datasets',
                 normalize:  dict(help='if set to true, row-normalizes features') = False,
                 features:   dict(help='features to use', choices=supported_features) = 'original',
                 sparse:     dict(help='if set to true, uses sparse matrices') = True,
                 ):

        self.name = dataset
        self.data_dir = data_dir
        self.normalize = normalize
        self.features = features
        self.sparse = sparse

    def load(self, verbose=False):
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data.edge_index, _ = remove_self_loops(data.edge_index)

        transforms = [RemoveIsolatedNodes(), RandomNodeSplit(split='train_rest')]
        transforms += [ToSparseTensor()] if self.sparse else []
        data = Compose(transforms)(data)

        if self.features.startswith('randproj'):
            dim = int(self.features[8:])
            transformer = GaussianRandomProjection(n_components=dim)
            x = transformer.fit_transform(data.x)
            data.x = torch.tensor(x, dtype=torch.float)

        if self.normalize:
            data.x = F.normalize(data.x, p=2., dim=-1)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data):
        if hasattr(data, 'adj_t'):
            nodes_degree = data.adj_t.sum(dim=1)
        else:
            nodes_degree = degree(data.edge_index[1], num_nodes=data.num_nodes)

        baseline = (data.y.unique(return_counts=True)[1].max().item() * 100 / data.num_nodes)
        train_ratio = data.train_mask.sum().item() / data.num_nodes * 100
        val_ratio = data.val_mask.sum().item() / data.num_nodes * 100
        test_ratio = data.test_mask.sum().item() / data.num_nodes * 100

        stat = {
            'name': self.name,
            'nodes': f'{data.num_nodes:,}',
            'edges': f'{data.num_edges:,}',
            'features': f'{data.num_features:,}',
            'classes': f'{int(data.y.max().item() + 1)}',
            'mean degree': f'{nodes_degree.mean().item():.2f}',
            'median degree': f'{nodes_degree.median().item()}',
            'train/val/test (%)': f'{train_ratio:.2f} / {val_ratio:.2f} / {test_ratio:.2f}',
            'baseline (%)': f'{baseline:.2f}'
        }

        highlighter = ReprHighlighter()
        table = Table(*stat.keys(), title="dataset stats", box=box.HORIZONTALS)
        table.add_row(*map(highlighter, stat.values()))
        console.log(table)
        console.print()
