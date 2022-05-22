import os
import pandas as pd
from pysrc.loggers.base import LoggerBase, if_enabled


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
