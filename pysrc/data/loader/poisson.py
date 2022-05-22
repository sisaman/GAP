import torch


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
