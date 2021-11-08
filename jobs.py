import os
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tabulate import tabulate
from tqdm import tqdm
from pathlib import Path


class JobManager:
    def __init__(self, args):
        self.args = args
        self.command = args.command
        self.file = args.file

    @property
    def path(self):
        return os.path.realpath(self.file)

    @property
    def dir(self):
        return os.path.dirname(self.path)

    @property
    def name(self):
        return Path(self.path).stem

    @property
    def output_dir(self):
        return os.path.join(self.dir, self.name)

    def run(self):
        if self.command == 'submit':
            self.submit()
        elif self.command == 'status':
            self.status()
        elif self.command == 'resubmit':
            self.resubmit()
        elif self.command == 'exec':
            self.exec()

    def submit(self):
        window = 7500
        num_cmds = sum(1 for _ in open(self.file))
        os.makedirs(self.output_dir, exist_ok=True)

        for i in tqdm(range(0, num_cmds, window), desc='submitting jobs'):
            begin = i + 1
            end = min(i + window, num_cmds)

            job_file_content = [
                f'#$ -N {self.name}-{begin}-{end}\n',
                f'#$ -S /bin/bash\n',
                f'#$ -P ai4media\n',
                f'#$ -l pytorch,sgpu,gpumem=24\n',
                f'#$ -t {begin}-{end}\n',
                f'#$ -o {self.output_dir}\n',
                f'#$ -e {self.output_dir}\n',
                f'#$ -cwd\n',
                f'#$ -V\n',
                f'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:10240\n',
                f'python jobs.py -f {self.file} exec --id $SGE_TASK_ID \n'
            ]

            job_file = os.path.join(self.output_dir, f'{self.name}-{begin}-{end}.job')

            with open(job_file, 'w') as file:
                file.writelines(job_file_content)
                file.flush()

            try:
                subprocess.check_call(['qsub', job_file], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print('\n\n', e.output, '\n\n')
                raise e

        print('done')

    def resubmit(self):
        failed_jobs = self.get_failed_jobs()

        if len(failed_jobs):
            with open(self.file) as jobs_file:
                job_list = jobs_file.read().splitlines()
            
            run_cmds = [job_list[i - 1] for i, _, _ in failed_jobs]

            with open(self.args.new_file, 'w') as file:
                for run in tqdm(run_cmds):
                    file.write(run + '\n')

            print('new job file created:', self.args.new_file)
            self.file = self.args.new_file
            self.submit()

    def status(self):
        failed_jobs = self.get_failed_jobs()
        print(tabulate(failed_jobs, headers=['job id', 'error file', 'num lines']) + '\n')

    def exec(self):
        with open(self.file) as jobs_file:
            job_list = jobs_file.read().splitlines()

        if self.args.all:
            for cmd in job_list:
                subprocess.check_call(cmd.split())
        else:
            subprocess.check_call(job_list[self.args.id - 1].split())

    def get_failed_jobs(self):
        file_list = [
            os.path.join(self.output_dir, file)
            for file in os.listdir(self.output_dir) if file.count('.e')
        ]

        failed_jobs = []
        for file in file_list:
            num_lines = sum(1 for _ in open(file))
            if num_lines > 0:
                job_id = int(file.split('.')[-1])
                failed_jobs.append([job_id, file, num_lines])

        return failed_jobs

    @staticmethod
    def register_arguments(parser):
        parser.add_argument('-f', '--file', type=str, required=True, help='jobs file name')
        command_subparser = parser.add_subparsers(dest='command')

        command_subparser.add_parser('submit')
        command_subparser.add_parser('status')

        parser_resubmit = command_subparser.add_parser('resubmit')
        parser_resubmit.add_argument('-n', '--new-file', type=str, help='name of new jobs file', required=True)

        parser_exec = command_subparser.add_parser('exec')
        parser_exec.add_argument('--id', type=int)
        parser_exec.add_argument('--all', action='store_true')

        return parser


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = JobManager.register_arguments(parser)
    args = parser.parse_args()
    print(args)

    JobManager(args).run()


if __name__ == '__main__':
    main()
