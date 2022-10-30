import os
import wandb
import pandas as pd
import numpy as np
from itertools import product


class WandBJobRegistry:
    """Job registry utility based on WandB

    Args:
        entity (str): Name of the entity (e.g., the username).
        project (str): Project name.
    """
    def __init__(self, entity, project):
        self.entity = entity
        self.project = project
        self.job_list = []
        self.df_jobs = pd.DataFrame()

    def pull(self):
        """Pull all runs from WandB"""
        api = wandb.Api()
        projects = [project.name for project in api.projects(entity=self.entity)]

        if self.project in projects:
            runs = api.runs(f"{self.entity}/{self.project}", per_page=2000)
            config_list = []
            for run in runs:
                config_list.append({k: v for k,v in run.config.items() if not k.startswith('_')})

            self.df_jobs = pd.DataFrame.from_dict(config_list)
            if 'epsilon' in self.df_jobs.columns:
                self.df_jobs['epsilon'] = self.df_jobs['epsilon'].astype(float)
    
    def register(self, main_file, *args, **params) -> list[str]:
        """Register jobs to the registry.
        This method will generate all possible combinations of the parameters and
        create a list of jobs to run. The job commands are stored in the registry
        if the corresponding runs are not already present in the WandB server.

        Args:
            main_file (str): Path to the main executable python file.
            *args (list): List of arguments to pass to the main file.
            **params (dict): Dictionary of parameters to sweep over.

        Returns:
            list[str]: List of jobs to run.
        """
        for key, value in params.items():
            if not (isinstance(value, list) or isinstance(value, tuple)):
                params[key] = (value,)
        
        jobs = []
        configs = self._product_dict(params)

        for config in configs:
            if not self._exists(config):
                self.df_jobs = pd.concat([self.df_jobs, pd.DataFrame(config, index=[0])], ignore_index=True)
                options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
                command = f'python {main_file} {" ".join(args)} {options} --logger wandb --project {self.project}'
                command = ' '.join(command.split())
                jobs.append(command)

        self.job_list += jobs
        return jobs

    def save(self, path: str, sort=False, shuffle=False):
        """Save the job list to a file.

        Args:
            path (str): Path to the file.
            sort (bool, optional): Sort the job list. Defaults to False.
            shuffle (bool, optional): Shuffle the job list. Defaults to False.
        """
        assert not (sort and shuffle), 'cannot sort and shuffle at the same time'

        if sort:
            jobs = sorted(self.job_list)
        elif shuffle:
            jobs = np.random.choice(self.job_list, len(self.job_list), replace=False)
        else:
            jobs = self.job_list

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            for item in jobs:
                print(item, file=file)

    def _exists(self, config: dict) -> bool:
        """Check if a run with the given config exists in the registry.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            bool: True if the run exists, False otherwise.
        """
        if set(config.keys()) - set(self.df_jobs.columns):
            # if config has a key not in df_jobs, return an empty df
            return False
        else:
            # find rows in df_jobs corresponding to runs that match config
            rows = self.df_jobs.loc[np.all([self.df_jobs[k] == v for k, v in config.items()], axis=0), :]
            return len(rows) > 0

    def _product_dict(self, params):
        """Generate all possible combinations of the parameters.
            
        Args:
            params (dict): Dictionary of parameters to sweep over.

        Yields:
            dict: Dictionary of individual parameters.
        """
        keys = params.keys()
        vals = params.values()
        for instance in product(*vals):
            yield dict(zip(keys, instance))