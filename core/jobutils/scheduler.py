import os
from pathlib import Path
import subprocess
import dask
from dask.distributed import Client
from dask_jobqueue import JobQueueCluster
from dask.distributed import as_completed
from rich.progress import Progress
from subprocess import CalledProcessError
from class_resolver import ClassResolver
from core import console, Console


cluster_resolver = ClassResolver.from_subclasses(JobQueueCluster, suffix='Cluster')


class JobScheduler:
    """Distributed job scheduler using Dask.

    Args:
        job_file (str): The job file to schedule. Each line is a job command.
        scheduler (str, optional): The scheduler to use. Options are 'htcondor', 
            'lsf', 'moab', 'oar', 'pbs', 'sge', 'slurm'. Defaults to 'sge'.
    """
    def __init__(self, job_file: str, scheduler: str = 'sge', config: dict = None):
        assert scheduler in cluster_resolver.options, f'Invalid scheduler: {scheduler}'
        self.scheduler = scheduler
        self.file = job_file
        path = os.path.realpath(self.file)
        self.name = Path(path).stem
        self.job_dir = os.path.join(os.path.dirname(path), self.name)

        if config:
            dask_config = dask.config.config
            updated_config = dask.config.merge(dask_config, config)
            dask.config.set(updated_config)

        with open(self.file) as jobs_file:
            self.job_list = jobs_file.read().splitlines()

    def submit(self):
        """Submit the job file to the cluster.

        Returns:
            list[str]: The list of failed job commands.
        """

        max_gpus = 160
        total = len(self.job_list)
        progress = SchedulerProgress(total=total, console=console)

        num_failed_jobs = 0
        failures_dir = os.path.join(self.job_dir, 'failures')
        os.makedirs(failures_dir, exist_ok=True)

        cluster: JobQueueCluster = cluster_resolver.make(
            self.scheduler,
            job_name=f'dask-{self.name}',
            log_directory=os.path.join(self.job_dir, 'logs'),
        )

        console.info(f'dashboard at {cluster.dashboard_link}')
        
        with cluster:   
            cluster.adapt(minimum=min(160, total))  # 160 is the max number of gpus on the cluster
            
            with Client(cluster) as client:
                with progress:
                    futures = client.map(
                        lambda cmd: subprocess.run(args=cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT),
                        self.job_list[:max_gpus],
                        pure=False,
                    )
                    submitted = min(max_gpus, total)
                    futures = as_completed(futures, with_results=False)
                    for future in futures:
                        try:
                            future.result()
                            progress.update(failed=False)
                        except CalledProcessError as e:
                            num_failed_jobs += 1
                            job_cmd = ' '.join(e.cmd)
                            failed_job_output = e.output.decode()
                            with open(os.path.join(failures_dir, f'{num_failed_jobs}.log'), 'w') as f:
                                print(job_cmd, end='\n\n', file=f)
                                print(failed_job_output, file=f)
                            progress.update(failed=True)
                        if submitted < total:
                            future = client.submit(
                                lambda cmd: subprocess.run(args=cmd.split(), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT),
                                self.job_list[submitted],
                                pure=False,
                            )
                            futures.add(future)
                            submitted += 1


class SchedulerProgress:
    """Progress bar for the scheduler.

    Args:
        total (int): The total number of jobs.
        console (Console): The rich console to use.
    """
    def __init__(self, total: int, console: Console):
        self.finished = 0
        self.failed = 0
        self.remaining = total

        self.bar = Progress(
            *Progress.get_default_columns(),
            "Completed: [green]{task.fields[finished]}",
            "Failed: [red]{task.fields[failed]}",
            "Remaining: [blue]{task.fields[remaining]}",
            console=console,
        )

        self.task = self.bar.add_task(
            description="Running jobs",
            finished=self.finished,
            failed=self.failed,
            remaining=self.remaining,
            total=total
        )

    def update(self, failed: bool):
        """Update the progress bar.

        Args:
            failed (bool): Whether the job failed or not.
        """
        self.finished += int(not failed)
        self.failed += int(failed)
        self.remaining -= 1

        self.bar.update(
            self.task,
            advance=1,
            finished=self.finished,
            failed=self.failed,
            remaining=self.remaining,
            refresh=True
        )

    def __enter__(self):
        return self.bar.__enter__()

    def __exit__(self, *args):
        return self.bar.__exit__(*args)
