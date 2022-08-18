from typing import Literal, Optional
from torch import Tensor
from torch.nn import Module
from torch.types import Number
from torch_geometric.data import Data
from abc import ABC, abstractmethod


Stage = Literal['train', 'val', 'test']
Metrics = dict[str, Number]


class TrainableModule(Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs): pass

    @abstractmethod
    def step(self, data: Data, stage: Stage) -> tuple[Optional[Tensor], Metrics]: pass

    @abstractmethod
    def predict(self, data: Data) -> Tensor: pass

    @abstractmethod
    def reset_parameters(self): pass
