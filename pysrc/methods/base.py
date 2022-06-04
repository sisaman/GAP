from abc import ABC, abstractmethod
from typing import Annotated, Optional
from torch import Tensor
from torch_geometric.data import Data
from pysrc.trainer import Trainer


class MethodBase(ABC):
    def __init__(self, 
                 num_classes: int,
                 device:  Annotated[str,   dict(help='device to use', choices=['cpu', 'cuda'])] = 'cuda',
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
    def fit(self, data: Data) -> dict[str, object]: pass

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor: pass
