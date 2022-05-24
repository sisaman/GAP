import os
import pandas as pd
from pysrc.loggers.base import LoggerBase, if_enabled


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

    @if_enabled
    def finish(self):
        os.makedirs(self.output_dir, exist_ok=True)
        df = pd.DataFrame(self.experiment, index=[0])
        df.to_csv(os.path.join(self.output_dir,
                  f'{self.experiment_id}.csv'), index=False)
