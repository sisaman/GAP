import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from collections.abc import Iterator


class PoissonDataLoader(DataLoader):
    def __init__(self, dataset: TensorDataset, batch_size: int):
        super().__init__(dataset, batch_size=batch_size)
        self.dataset_size = len(dataset)
        self.prob = batch_size / self.dataset_size
        self.sampler_mask: torch.Tensor = torch.empty(
            self.dataset_size, 
            dtype=bool, 
            device=dataset.tensors[0].device
        )
        self.sampler_mask.bernoulli_(self.prob).bool()

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for _ in range(0, self.dataset_size, self.batch_size):
            yield self.dataset[self.sampler_mask]
            self.sampler_mask.bernoulli_(self.prob).bool()
            
    def __len__(self) -> int:
        return self.dataset_size // self.batch_size
