import os
import logging
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm


class JobManager:
    def __init__(self, args, cmd_generator=None):
        self.args = args
        self.name = args.name
        self.command = args.command
        self.jobs_dir = args.jobs_dir
        self.cmd_generator = cmd_generator

    def run(self):
        if self.command == 'create':
            self.create()
        elif self.command == 'submit':
            self.submit()
        elif self.command == 'status':
            self.status()
        elif self.command == 'resubmit':
            self.resubmit()
        elif self.command == 'exec':
            self.exec()

    def create(self):
        os.makedirs(self.jobs_dir, exist_ok=True)
        run_cmds = self.cmd_generator(self.args)

        with open(os.path.join(self.jobs_dir, f'{self.name}.jobs'), 'w') as file:
            for run in tqdm(run_cmds):
                file.write(run + '\n')

        print('job file created:', os.path.join(self.jobs_dir, f'{self.name}.jobs'))

    def submit(self):
        window = 7500
        num_cmds = sum(1 for _ in open(os.path.join(self.jobs_dir, f'{self.name}.jobs')))

        for i in tqdm(range(0, num_cmds, window), desc='submitting jobs'):
            begin = i + 1
            end = min(i + window, num_cmds)

            job_file_content = [
                f'#$ -N {self.name}-{begin}-{end}\n',
                f'#$ -S /bin/bash\n',
                f'#$ -P ai4media\n',
                f'#$ -l pytorch,sgpu,gpumem=24\n',
                f'#$ -t {begin}-{end}\n',
                f'#$ -o {self.jobs_dir}\n',
                f'#$ -e {self.jobs_dir}\n',
                f'#$ -cwd\n',
                f'#$ -V\n',
                f'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:10240\n',
                f'python jobs.py -n {self.name} exec --id $SGE_TASK_ID \n'
            ]

            file_name = os.path.join(self.jobs_dir, f'{self.name}-{begin}-{end}.job')

            with open(file_name, 'w') as file:
                file.writelines(job_file_content)
                file.flush()

            try:
                subprocess.check_call(['qsub', file_name], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print('\n\n', e.output, '\n\n')
                raise e

        print('done')

    def resubmit(self):
        failed_jobs = self.get_failed_jobs()

        if len(failed_jobs):
            with open(os.path.join(self.jobs_dir, f'{self.name}.jobs')) as jobs_file:
                job_list = jobs_file.read().splitlines()

            self.cmd_generator = lambda _: [job_list[i - 1] for i, _, _ in failed_jobs]
            self.name = f'{self.name}-resubmit'
            self.create()
            self.submit()

    def status(self):
        try:
            import tabulate
        except ImportError:
            tabulate = None

        failed_jobs = self.get_failed_jobs()

        if tabulate:
            print(tabulate.tabulate(failed_jobs, headers=['job id', 'error file', 'num lines']))
        else:
            for _, file, num_lines in failed_jobs:
                print(num_lines, os.path.join(self.jobs_dir, file))

        print()

    def exec(self):
        with open(os.path.join(self.jobs_dir, f'{self.name}.jobs')) as jobs_file:
            job_list = jobs_file.read().splitlines()

        if self.args.all:
            for cmd in job_list:
                subprocess.check_call(cmd.split())
        else:
            subprocess.check_call(job_list[self.args.id - 1].split())

    def get_failed_jobs(self):
        failed_jobs = []
        file_list = [
            os.path.join(self.jobs_dir, file)
            for file in os.listdir(self.jobs_dir) if file.startswith(self.name) and file.count('.e')
        ]

        for file in file_list:
            num_lines = sum(1 for _ in open(file))
            if num_lines > 0:
                job_id = int(file.split('.')[-1])
                failed_jobs.append([job_id, file, num_lines])

        return failed_jobs

    @staticmethod
    def register_arguments(parser, default_jobs_dir='./jobs'):
        parser.add_argument('-n', '--name', type=str, required=True, help='experiment name')
        parser.add_argument('-j', '--jobs-dir', type=str, default=default_jobs_dir, help='jobs directory')
        command_subparser = parser.add_subparsers(dest='command')

        parser_create = command_subparser.add_parser('create')
        command_subparser.add_parser('submit')
        command_subparser.add_parser('status')
        command_subparser.add_parser('resubmit')

        parser_exec = command_subparser.add_parser('exec')
        parser_exec.add_argument('--id', type=int)
        parser_exec.add_argument('--all', action='store_true')

        return parser, parser_create


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser, parser_create = JobManager.register_arguments(parser)
    args = parser.parse_args()
    print(args)

    JobManager(args).run()


if __name__ == '__main__':
    main()
