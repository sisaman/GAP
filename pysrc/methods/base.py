from abc import ABC, abstractmethod
from typing import Optional
from torch import Tensor
from torch_geometric.data import Data

class MethodBase(ABC):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @abstractmethod
    def reset_parameters(self) -> None: pass

    @abstractmethod
    def fit(self, data: Data) -> dict[str, object]: pass

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor: pass
