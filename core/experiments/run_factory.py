import os
import wandb
import pandas as pd
import numpy as np
from itertools import product


class RunFactory:
    def __init__(self, entity, project, check_existing=True):
        self.project = project
        self.check_existing = check_existing
        self.cmd_list = []

        if check_existing:
            api = wandb.Api()
            projects = [project.name for project in api.projects(entity=entity)]
            if project not in projects:
                self.runs_df = pd.DataFrame()
            else:
                runs = api.runs(f"{entity}/{project}", per_page=2000)
                config_list = []
                for run in runs:
                    config_list.append({k: v for k,v in run.config.items() if not k.startswith('_')})

                self.runs_df = pd.DataFrame.from_dict(config_list)
                if 'epsilon' in self.runs_df.columns:
                    self.runs_df['epsilon'] = self.runs_df['epsilon'].astype(float)
    
    def register(self, method: str, **params) -> list[str]:
        for key, value in params.items():
            if not (isinstance(value, list) or isinstance(value, tuple)):
                params[key] = (value,)
        
        cmd_list = []
        configs = self.product_dict(params)

        for config in configs:
            if not self.check_existing or len(self.find_runs(config)) == 0:
                self.runs_df = pd.concat([self.runs_df, pd.DataFrame(config, index=[0])], ignore_index=True)
                options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
                command = f'python train.py {method} {options} --logger wandb --project {self.project}'
                command = ' '.join(command.split())
                cmd_list.append(command)

        self.cmd_list += cmd_list
        return cmd_list

    def get_all_runs(self) -> list[str]:
        return self.cmd_list

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            for item in self.cmd_list:
                print(item, file=file)
        print(f'Saved {len(self.cmd_list)} runs to {path}')

    def find_runs(self, config: dict) -> pd.DataFrame:
        if set(config.keys()) - set(self.runs_df.columns):
            # if config has a key not in runs_df, return an empty df
            return pd.DataFrame()
        else:
            # return a df with rows corresponding to runs that match config
            return self.runs_df.loc[np.all([self.runs_df[k] == v for k, v in config.items()], axis=0), :]

    @staticmethod
    def product_dict(params):
        keys = params.keys()
        vals = params.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))