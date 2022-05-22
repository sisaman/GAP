import os
from pysrc.loggers.base import LoggerBase, if_enabled


try:
    import wandb
except ImportError:
    wandb = None


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
