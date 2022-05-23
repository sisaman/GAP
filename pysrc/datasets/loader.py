import os
from functools import partial
from rich.highlighter import ReprHighlighter
from rich import box
from pysrc.console import console
from rich.table import Table
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import Compose, RemoveIsolatedNodes, ToSparseTensor, RandomNodeSplit
from pysrc.data.transforms import FilterClassByCount
from pysrc.data.transforms import RemoveSelfLoops
from pysrc.datasets import Facebook
from pysrc.datasets import Amazon


class DatasetLoader:
    supported_datasets = {
        # main datasets
        'reddit': partial(Reddit, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'amazon': partial(Amazon, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'facebook': partial(Facebook, name='UIllinois20', target='year', 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=1000, remove_unlabeled=True)
            ])
        ),
    }

    def __init__(self,
                 dataset:    dict(help='name of the dataset', choices=supported_datasets) = 'facebook',
                 data_dir:   dict(help='directory to store the dataset') = './datasets',
                 ):

        self.name = dataset
        self.data_dir = data_dir

    def load(self, verbose=False) -> Data:
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
