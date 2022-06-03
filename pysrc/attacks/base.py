from typing import Annotated
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from pysrc.classifiers.base import Stage
from pysrc.methods.mlp import MLP


class AttackModelBase(MLP, ABC):
    def __init__(self, 
                 max_samples: Annotated[int,   dict(help='maximum number of attack dataset samples')] = 10000,
                 val_ratio:   Annotated[float, dict(help='ratio of validation set size')] = 0.2,
                 test_ratio:  Annotated[float, dict(help='ratio of test set size')] = 0.3,
                 **kwargs:    Annotated[dict,  dict(help='extra options passed to base class', bases=[MLP])]
                ):

        super().__init__(
            num_classes=2,  # either member or non-member
            **kwargs
        )

        self.max_samples = max_samples
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    @abstractmethod
    def prepare_attack_dataset(self, data: Data) -> tuple[Tensor, Tensor]: pass

    def split(self, num_samples: int) -> tuple[Tensor, Tensor, Tensor]:
        num_train = int(num_samples * (1 - self.val_ratio - self.test_ratio))
        num_val = int(num_samples * self.val_ratio)
        
        perm = torch.randperm(num_samples, device=self.device)
        train_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        val_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        train_mask[perm[:num_train]] = True
        val_mask[perm[num_train:num_train + num_val]] = True
        test_mask[perm[num_train + num_val:]] = True

        return train_mask, val_mask, test_mask

    def data_loader(self, stage: Stage) -> DataLoader:
        if not hasattr(self.data, 'attack_data'):
            x, y = self.prepare_attack_dataset(self.data)
            x = torch.stack([x], dim=-1)
            train_mask, val_mask, test_mask = self.split(y.size(0))
            self.data.attack_data = {
                'train': (x[train_mask], y[train_mask]),
                'val': (x[val_mask], y[val_mask]),
                'test': (x[test_mask], y[test_mask])
            }

        x, y = self.data.attack_data[stage]
        if self.batch_size == 'full' or (stage != 'train' and self.full_batch_eval):
            return [(x, y)]
        else:
            return DataLoader(
                dataset=TensorDataset(x, y),
                batch_size=self.batch_size, 
                shuffle=True
            )