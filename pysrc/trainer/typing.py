from typing import Literal
from torch.types import Number


TrainerStage = Literal['train', 'val', 'test']
Metrics = dict[str, Number]
