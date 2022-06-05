from abc import ABC, abstractmethod
from typing import Annotated, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from pysrc.classifiers.base import Metrics
from pysrc.trainer import Trainer

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MethodBase(ABC):
    def __init__(self, 
                 num_classes: int,
                 device:  Annotated[str,   dict(help='device to use', choices=['cpu', 'cuda'])] = default_device,
                 use_amp: Annotated[bool,  dict(help='use automatic mixed precision training')] = False,
                 ):

        self.num_classes = num_classes
        self.device = device
        self.use_amp = use_amp

        self.trainer = Trainer(
            use_amp=self.use_amp, 
            monitor='val/acc', monitor_mode='max', 
            device=self.device,
        )

    def reset_parameters(self) -> None:
        self.trainer.reset()

    @abstractmethod
    def fit(self, data: Data) -> Metrics:
        """Fit the model to the given data."""

    @abstractmethod
    def test(self, data: Optional[Data] = None) -> Metrics:
        """Test the model on the given data, or the training data if data is None."""

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
