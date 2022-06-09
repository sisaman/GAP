from abc import ABC, abstractmethod
from typing import Annotated, Literal, Optional
from torch import Tensor
from torch_geometric.data import Data
from core.classifiers.base import Metrics
from core.trainer import Trainer


class MethodBase(ABC):
    def __init__(self, 
                 num_classes: int, 
                 device:  Literal['cpu', 'cuda'] = 'cuda', 
                 use_amp: bool = False,
                 val_interval: Annotated[int, dict(help='interval of validation')] = 1,
                 ):
        self.num_classes = num_classes
        self.device = device
        self.use_amp = use_amp

        self.trainer = Trainer(
            val_interval=val_interval,
            use_amp=self.use_amp, 
            monitor='val/acc', 
            monitor_mode='max', 
            device=self.device,
        )

    def reset_parameters(self) -> None:
        self.trainer.reset()

    @abstractmethod
    def fit(self, data: Data, prefix: str = '') -> Metrics:
        """Fit the model to the given data."""

    @abstractmethod
    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        """Test the model on the given data, or the training data if data is None."""

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""