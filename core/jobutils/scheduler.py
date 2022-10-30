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

dask.config.set({"jobqueue.sge.walltime": None})
dask.config.set({"distributed.worker.memory.target": False})    # Avoid spilling to disk
dask.config.set({"distributed.worker.memory.spill": False})     # Avoid spilling to disk
dask.config.set({'distributed.scheduler.allowed-failures': 99}) # Allow workers to fail


class JobScheduler:
    """Distributed job scheduler using Dask.

    Args:
        job_file (str): The job file to schedule. Each line is a job command.
        scheduler (str, optional): The scheduler to use. Options are 'htcondor', 
            'lsf', 'moab', 'oar', 'pbs', 'sge', 'slurm'. Defaults to 'sge'.
    """
    def __init__(self, job_file: str, scheduler: str = 'sge'):
        assert scheduler in self.cluster_resolver.options, f'Invalid scheduler: {scheduler}'
        self.scheduler = scheduler
        self.file = job_file
        path = os.path.realpath(self.file)
        self.name = Path(path).stem
        self.job_dir = os.path.join(os.path.dirname(path), self.name)

        with open(self.file) as jobs_file:
            self.job_list = jobs_file.read().splitlines()

    def submit(self):
        """Submit the job file to the cluster.

        Returns:
            list[str]: The list of failed job commands.
        """

        cluster = self.cluster_resolver.make(
            self.scheduler,
            job_name=f'dask-{self.name}',
            log_directory=os.path.join(self.job_dir, 'logs'),
        )
        
        with cluster:
            total = len(self.job_list)
            cluster.adapt(minimum=min(100, total))
            
            with Client(cluster) as client:
                futures = client.map(self.execute, range(1, total + 1))
                progress = SchedulerProgress(total=total, console=console)
                
                failed_jobs = {}
                failures_dir = os.path.join(self.job_dir, 'failures')
                os.makedirs(failures_dir, exist_ok=True)

                try:
                    with progress:
                        for future in as_completed(futures, with_results=False):
                            try:
                                future.result()
                                progress.update(failed=False)
                            except CalledProcessError as e:
                                job_cmd = ' '.join(e.cmd)
                                failed_jobs[job_cmd] = e.output.decode()
                                with open(os.path.join(failures_dir, f'{len(failed_jobs)}.log'), 'w') as f:
                                    print(job_cmd, end='\n\n', file=f)
                                    print(failed_jobs[job_cmd], file=f)
                                progress.update(failed=True)
                except KeyboardInterrupt:
                    console.warning('Graceful Shutdown')

                return failed_jobs

    def execute(self, line: int = None):
        """Execute all or a single line of the job file.

        Args:
            line (int, optional): The line to execute (starting from 1). 
                Defaults to None, which runs all the lines.
        
        Raises:
            CalledProcessError: If the command returns a non-zero exit status.
        """

        if line is None:
            for cmd in self.job_list:
                subprocess.run(cmd.split())
        else:
            subprocess.run(
                args=self.job_list[line - 1].split(), 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT
            )

    cluster_resolver = ClassResolver.from_subclasses(JobQueueCluster, suffix='Cluster')



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
