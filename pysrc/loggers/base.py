import functools
import uuid
from argparse import Namespace


def if_enabled(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.enabled:
            return func(self, *args, **kwargs)
    return wrapper


class LoggerBase:
    def __init__(self, project=None, output_dir='./output', experiment_id=str(uuid.uuid1()),
                 enabled=True, config=Namespace()):
        self.project = project
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        self.enabled = enabled
        self.config = config
        self.config.experiment_id = experiment_id
        self._experiment = None

    def log(self, metrics): pass
    def log_summary(self, metrics): pass
    def watch(self, model): pass
    def finish(self): pass
    def enable(self): self.enabled = True
    def disable(self): self.enabled = False

    @property
    def experiment(self):
        return self._experiment
