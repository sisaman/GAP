import functools
from typing import Callable
from uuid import uuid1
from abc import ABC, abstractmethod
from torch.nn import Module
from pysrc.utils import RT


def if_enabled(func: Callable[..., RT]) -> Callable[..., RT]:
    @functools.wraps(func)
    def wrapper(self: LoggerBase, *args, **kwargs) -> RT:
        if self.enabled:
            return func(self, *args, **kwargs)
    return wrapper


class LoggerBase(ABC):
    def __init__(self, project: str=None, output_dir: str='./output', 
                 experiment_id: str=str(uuid1()), enabled: bool=True, 
                 config: dict={}):
        self.project = project
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        self.enabled = enabled
        self.config = config
        self.config[experiment_id] = experiment_id

    @property
    @abstractmethod
    def experiment(self): pass

    @abstractmethod
    def log(self, metrics: dict[str, object]): pass
    
    @abstractmethod
    def log_summary(self, metrics: dict[str, object]): pass

    @abstractmethod
    def watch(self, model: Module, **kwargs): pass
    
    @abstractmethod
    def finish(self): pass

    def enable(self): 
        self.enabled = True

    def disable(self): 
        self.enabled = False
