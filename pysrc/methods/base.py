from abc import ABC, abstractmethod
from torch_geometric.data import Data

class MethodBase(ABC):
    @abstractmethod
    def reset_parameters(self) -> None: pass

    @abstractmethod
    def fit(self, data: Data) -> dict[str, object]: pass
