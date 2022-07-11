import os

from torch.nn import Module
from core.loggers.base import LoggerBase, if_enabled

try: 
    import wandb
    from wandb.wandb_run import Run
except ImportError: 
    wandb = None


class WandbLogger(LoggerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if wandb is None:
            raise ImportError('wandb is not installed yet, install it with `pip install wandb`.')

    @property
    def experiment(self) -> Run:
        if not hasattr(self, '_experiment'):
            os.environ["WANDB_SILENT"] = "true"
            settings = wandb.Settings(start_method="fork")
            os.makedirs(self.output_dir, exist_ok=True)

            self._experiment = wandb.init(
                project=self.project,
                dir=self.output_dir,
                reinit=True, 
                resume='allow', 
                config=self.config, 
                save_code=True,
                settings=settings
            )

        return self._experiment

    @if_enabled
    def log(self, metrics: dict[str, object]):
        self.experiment.log(metrics)

    @if_enabled
    def log_summary(self, metrics: dict[str, object]):
        for metric, value in metrics.items():
            self.experiment.summary[metric] = value

    @if_enabled
    def watch(self, model: Module, **kwargs):
        self.experiment.watch(model, **kwargs)

    @if_enabled
    def finish(self):
        self.experiment.finish()
