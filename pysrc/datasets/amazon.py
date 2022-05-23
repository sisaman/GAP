import torch
from ogb.nodeproppred import PygNodePropPredDataset


class Amazon(PygNodePropPredDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Amazon, self).__init__(root, name='ogbn-products', transform=None, pre_transform=pre_transform)
        self.data.y = self.data.y.view(-1)
        split_idx = self.get_idx_split()
        for split, idx in split_idx.items():
            mask = torch.zeros(self.data.num_nodes, dtype=bool).scatter(0, idx, True)
            split = 'val' if split == 'valid' else split
            self.data[f'{split}_mask'] = mask
        if transform:
            self.data = transform(self.data)
