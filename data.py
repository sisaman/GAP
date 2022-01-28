import os
import ssl
from functools import partial
from rich.highlighter import ReprHighlighter
from rich import box
from console import console
from rich.table import Table
import torch
from torch_geometric.utils import remove_self_loops, subgraph, from_scipy_sparse_matrix
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import Compose, BaseTransform, RemoveIsolatedNodes, ToSparseTensor, RandomNodeSplit
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
from scipy.io import loadmat
from torch_geometric.data import Data, InMemoryDataset, download_url
from sklearn.preprocessing import LabelEncoder
from torch_sparse import SparseTensor
from args import support_args
import torch.utils.cpp_extension


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


class NeighborSampler(BaseTransform):
    def __init__(self, max_degree: int):
        self.max_deg = max_degree
        
        try:
            edge_sampler = torch.ops.my_ops.sample_edge
        except RuntimeError:
            torch.utils.cpp_extension.load(
                name="sample",
                sources=['csrc/sample.cpp'],
                build_directory='csrc',
                is_python_module=False,
                verbose=True,
            )
            edge_sampler = torch.ops.my_ops.sample_edge
        
        self.edge_sampler = edge_sampler

    def __call__(self, data):
        N = data.num_nodes
        E = data.num_edges
        adj = data.adj_t.t()
        row, col, _ = adj.coo()
        perm = torch.randperm(E)
        row, col = row[perm], col[perm]
        row, col = self.edge_sampler(row.tolist(), col.tolist(), N, self.max_deg)
        adj = SparseTensor(row=row, col=col)
        data.adj_t = adj.t()


class FilterClassByCount(BaseTransform):
    def __init__(self, min_count, remove_unlabeled=False):
        self.min_count = min_count
        self.remove_unlabeled = remove_unlabeled

    def __call__(self, data):
        assert hasattr(data, 'y') and hasattr(data, 'train_mask')

        y = torch.nn.functional.one_hot(data.y)
        counts = y.sum(dim=0)
        y = y[:, counts >= self.min_count]
        mask = y.sum(dim=1).bool()        # nodes to keep
        data.y = y.argmax(dim=1)

        if self.remove_unlabeled:
            data.x = data.x[mask]
            data.y = data.y[mask]
            data.train_mask = data.train_mask[mask]
            data.val_mask = data.val_mask[mask]
            data.test_mask = data.test_mask[mask]
            data.edge_index, _ = subgraph(subset=mask, edge_index=data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        else:
            data.y[~mask] = -1                # set filtered nodes as unlabeled
            data.train_mask = data.train_mask & mask
            data.val_mask = data.val_mask & mask
            data.test_mask = data.test_mask & mask

        return data


class RemoveSelfLoops(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            data.edge_index, _ = remove_self_loops(data.edge_index)
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.remove_diag()
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


class PoissonDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.prob = batch_size / self.dataset_size
        self.sampler = torch.empty(
            self.dataset_size, 
            dtype=bool, 
            device=dataset.tensors[0].device
        ).bernoulli_(self.prob).bool()

    def __iter__(self):
        for _ in range(0, self.dataset_size, self.batch_size):
            yield self.dataset[self.sampler]
            self.sampler.bernoulli_(self.prob).bool()
            
    def __len__(self):
        return self.dataset_size // self.batch_size


@support_args
class Dataset:
    supported_datasets = {
        # main datasets
        'reddit': partial(Reddit, 
            transform=Compose([RandomNodeSplit(num_val=0.05, num_test=0.1), FilterClassByCount(min_count=10000, remove_unlabeled=True)])
        ),
        'amazon': partial(load_ogb, name='ogbn-products', 
            transform=Compose([RandomNodeSplit(num_val=0.05, num_test=0.1), FilterClassByCount(min_count=100000, remove_unlabeled=True)])
        ),
        'facebook': partial(Facebook100, name='UIllinois20', target='year', 
            transform=Compose([RandomNodeSplit(num_val=0.05, num_test=0.1), FilterClassByCount(min_count=1000, remove_unlabeled=True)])
        ),
    }

    supported_features = {'original', 'randproj16', 'randproj32', 'randproj64'}

    def __init__(self,
                 dataset:    dict(help='name of the dataset', choices=supported_datasets) = 'facebook',
                 data_dir:   dict(help='directory to store the dataset') = './datasets',
                 ):

        self.name = dataset
        self.data_dir = data_dir

    def load(self, verbose=False):
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor()])(data)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data):
        nodes_degree = data.adj_t.sum(dim=1)
        baseline = (data.y[data.test_mask].unique(return_counts=True)[1].max().item() * 100 / data.test_mask.sum().item())
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
