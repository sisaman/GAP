import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class RandomDataSplit(BaseTransform):
    def __init__(self, num_nodes_per_class, train_ratio=0.7, test_ratio=0.2):
        self.num_nodes_per_class = num_nodes_per_class
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def __call__(self, data: Data) -> Data:
        y = data.y
        num_classes = y.max().item() + 1
        num_train_nodes_per_class = int(self.num_nodes_per_class * self.train_ratio)
        num_test_nodes_per_class = int(self.num_nodes_per_class * self.test_ratio)

        train_mask = torch.zeros_like(y, dtype=torch.bool)
        test_mask = torch.zeros_like(y, dtype=torch.bool)
        val_mask = torch.zeros_like(y, dtype=torch.bool)
        
        for c in range(num_classes):
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            num_nodes = idx.size(0)
            if num_nodes >= self.num_nodes_per_class:    
                idx = idx[torch.randperm(idx.size(0))][:self.num_nodes_per_class]
                train_mask[idx[:num_train_nodes_per_class]] = True
                test_mask[idx[num_train_nodes_per_class:num_train_nodes_per_class + num_test_nodes_per_class]] = True
                val_mask[idx[num_train_nodes_per_class + num_test_nodes_per_class:]] = True

        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask

        return data