import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch

from args import from_args, add_parameters_as_argument, print_args, str2bool
from datasets import load_dataset
from loggers import Logger
from models import NodeClassifier
from trainer import Trainer
from utils import measure_runtime, colored_text, seed_everything, confidence_interval


@measure_runtime
def run(args):
    dataset = from_args(load_dataset, args)

    test_acc = []
    run_metrics = {}
    logger = from_args(Logger.create, args, enabled=args.debug, config=args)

    for iteration in range(args.repeats):
        g = dataset.clone().to('cpu' if args.cpu else 'cuda')
        model = from_args(NodeClassifier, args, input_dim=g.num_features, num_classes=g.num_classes)
        trainer = from_args(Trainer, args)
        best_metrics = trainer.fit(model, g)

        # process results
        for metric, value in best_metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

        test_acc.append(best_metrics['test/acc'])
        print('\nrun: %d\ntest/acc: %.2f\naverage: %.2f\n\n' % (iteration+1, test_acc[-1], np.mean(test_acc).item()))

    logger.enable()
    summary = {}
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=args.seed)
        logger.log_summary(summary)


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset)

    # model args
    group_model = init_parser.add_argument_group('model arguments')
    add_parameters_as_argument(NodeClassifier, group_model)

    # trainer arguments
    group_trainer = init_parser.add_argument_group('trainer arguments')
    add_parameters_as_argument(Trainer, group_trainer)
    group_trainer.add_argument('--cpu', help='train on CPU', type=str2bool, nargs='?', const=True, default=not torch.cuda.is_available())

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help="number of times the experiment is repeated")
    add_parameters_as_argument(Logger.create, group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    if not args.cpu and not torch.cuda.is_available():
        print(colored_text('CUDA is not available, falling back to CPU', color='red'))
        args.cpu = True

    try:
        run(args)
    except KeyboardInterrupt:
        print('Graceful Shutdown...')


if __name__ == '__main__':
    main()
