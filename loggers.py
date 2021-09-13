import functools
import os
import uuid
from argparse import Namespace
from args import support_args
import pandas as pd

try:
    import wandb
except ImportError:
    wandb = None


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


class WandbLogger(LoggerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if wandb is None:
            raise ImportError(
                'wandb is not installed yet, install it with `pip install wandb`.')

    @property
    def experiment(self):
        if self._experiment is None:
            os.environ["WANDB_SILENT"] = "true"
            settings = wandb.Settings(start_method="fork")  # noqa

            self._experiment = wandb.init(
                project=self.project,
                reinit=True, resume='allow', config=self.config, save_code=True,
                settings=settings)

        return self._experiment

    @if_enabled
    def log(self, metrics):
        self.experiment.log(metrics)

    @if_enabled
    def log_summary(self, metrics):
        for metric, value in metrics.items():
            self.experiment.summary[metric] = value

    @if_enabled
    def watch(self, model):
        self.experiment.watch(model, log_freq=50)

    @if_enabled
    def finish(self):
        self.experiment.finish()


class CSVLogger(LoggerBase):
    @property
    def experiment(self):
        if self._experiment is None:
            self._experiment = vars(self.config)
        return self._experiment

    @if_enabled
    def log_summary(self, metrics):
        self.experiment.update(metrics)

    @if_enabled
    def finish(self):
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(self.experiment, index=[0])
        df.to_csv(os.path.join(self.output_dir,
                  f'{self.experiment_id}.csv'), index=False)


@support_args
class Logger:
    instance = None

    @classmethod
    def get_instance(cls):
        return cls.instance

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.__init__(cls, *args, **kwargs)
        return cls.instance

    def __init__(self,
                 logger:        dict(help='select logger type', choices=['wandb', 'csv']) = 'csv',
                 project:       dict(help="project name for logger") = 'PrivateGNN',
                 output_dir:    dict(help="directory to store the results", option='-o') = './output',
                 debug:         dict(help='enable debugger logging') = False,
                 enabled=True,
                 config=Namespace()):
        LoggerCls = WandbLogger if debug or logger == 'wandb' else CSVLogger
        Logger.instance = LoggerCls(project=project, output_dir=output_dir, enabled=enabled, config=config)
