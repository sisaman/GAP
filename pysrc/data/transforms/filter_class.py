import torch
from torch_geometric.transforms import BaseTransform


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
            data = data.subgraph(mask)
        else:
            data.y[~mask] = -1                # set filtered nodes as unlabeled
            data.train_mask = data.train_mask & mask
            data.val_mask = data.val_mask & mask
            data.test_mask = data.test_mask & mask

        return data