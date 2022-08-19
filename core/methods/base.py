from abc import ABC, abstractmethod
from typing import Optional
from torch import Tensor
from torch_geometric.data import Data
from core.modules.base import Metrics


class MethodBase(ABC):

    @abstractmethod
    def fit(self, data: Data, prefix: str = '') -> Metrics:
        """Fit the model to the given data."""

    @abstractmethod
    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        """Test the model on the given data, or the training data if data is None."""

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
