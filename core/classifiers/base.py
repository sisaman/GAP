from abc import ABC, abstractmethod
from typing import Literal
from torch import Tensor
from torch.nn import Module
from torch.types import Number

Stage = Literal['train', 'val', 'test']
Metrics = dict[str, Number]

class ClassifierBase(Module, ABC):
    @abstractmethod
    def step(self, batch: tuple[Tensor, Tensor], stage: Stage) -> tuple[Tensor, Metrics]: pass
    @abstractmethod
    def predict(self, x: Tensor) -> Tensor: pass
    @abstractmethod
    def reset_parameters(self): pass
