import os
import pandas as pd
from torch.nn import Module
from core.loggers.base import LoggerBase, if_enabled


class CSVLogger(LoggerBase):
    @property
    def experiment(self) -> dict[str, object]:
        if not hasattr(self, '_experiment'):
            self._experiment = self.config
        return self._experiment

    @if_enabled
    def log(self, metrics: dict[str, object]):
        self.experiment.update(metrics)

    @if_enabled
    def log_summary(self, metrics: dict[str, object]):
        self.experiment.update(metrics)

    def watch(self, model: Module, **kwargs):
        pass

    @if_enabled
    def finish(self):
        log_dir = os.path.join(self.output_dir, 'csv', self.project)
        os.makedirs(log_dir, exist_ok=True)
        df = pd.DataFrame(self.experiment, index=[0])
        df.to_csv(os.path.join(log_dir, f'{self.experiment_id}.csv'), index=False)
