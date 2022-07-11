from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from core.experiments import JobManager


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = JobManager.register_arguments(parser)
    args = parser.parse_args()
    # print_args(args, num_cols=1)

    JobManager(args).run()


if __name__ == '__main__':
    main()
