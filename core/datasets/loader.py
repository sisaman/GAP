import os
from functools import partial
from typing import Annotated
import torch
from core import console
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import Compose, ToSparseTensor, RandomNodeSplit
from core.args.utils import ArgInfo
from core.data.transforms import FilterClassByCount
from core.data.transforms import RemoveSelfLoops
from core.data.transforms import RemoveIsolatedNodes
from core.datasets import Facebook
from core.datasets import Amazon
from core.utils import dict2table


class DatasetLoader:
    supported_datasets = {
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
                 dataset:    Annotated[str, ArgInfo(help='name of the dataset', choices=supported_datasets)] = 'facebook',
                 data_dir:   Annotated[str, ArgInfo(help='directory to store the dataset')] = './datasets',
                 ):

        self.name = dataset
        self.data_dir = data_dir

    def load(self, verbose=False) -> Data:
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor()])(data)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data: Data):
        nodes_degree: torch.Tensor = data.adj_t.sum(dim=1)
        baseline: float = (data.y[data.test_mask].unique(return_counts=True)[1].max().item() * 100 / data.test_mask.sum().item())
        train_ratio: float = data.train_mask.sum().item() / data.num_nodes * 100
        val_ratio: float = data.val_mask.sum().item() / data.num_nodes * 100
        test_ratio: float = data.test_mask.sum().item() / data.num_nodes * 100

        stat = {
            'nodes': f'{data.num_nodes:,}',
            'edges': f'{data.num_edges:,}',
            'features': f'{data.num_features:,}',
            'classes': f'{int(data.y.max() + 1)}',
            'mean degree': f'{nodes_degree.mean():.2f}',
            'median degree': f'{nodes_degree.median()}',
            'train/val/test (%)': f'{train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}',
            'baseline acc (%)': f'{baseline:.2f}'
        }

        table = dict2table(stat, num_cols=2, title=f'dataset: [yellow]{self.name}[/yellow]')
        console.info(table)
        console.print()
