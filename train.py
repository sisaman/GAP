import time
from console import console
with console.status('importing modules'):
    import logging
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    import torch
    from args import print_args
    from datasets import Dataset
    from loggers import Logger
    from methods import GAP
    from utils import seed_everything, confidence_interval
    from torch_geometric.data import Data


def run(args):
    with console.status('loading dataset'):
        data_initial = Dataset.from_args(args).load(verbose=True)

    test_acc = []
    run_metrics = {}
    num_classes = data_initial.y.max().item() + 1
    logger = Logger.from_args(args, enabled=args.debug, config=args)

    ### initiallize model ###
    model: GAP = GAP.from_args(args, num_classes=num_classes)

    ### run experiment ###
    for iteration in range(args.repeats):
        data = Data(**data_initial.to_dict())

        model.reset_parameters()
        best_metrics = model.fit(data)

        ### process results ###
        for metric, value in best_metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

        test_acc.append(best_metrics['test/acc'])
        console.print()
        logging.info(f'run: {iteration + 1}\t test/acc: {test_acc[-1]:.2f}\t average: {np.mean(test_acc).item():.2f}\n')

    logger.enable()
    summary = {}
    
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=args.seed)
        logger.log_summary(summary)

    logger.finish()
    print()


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    Dataset.add_args(group_dataset)

    # model args
    group_model = init_parser.add_argument_group('model arguments')
    GAP.add_args(group_model)

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-n', '--name', type=str, default=None, help='experiment name')
    group_expr.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help="number of times the experiment is repeated")
    Logger.add_args(group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    print_args(args, num_cols=4)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed:
        seed_everything(args.seed)

    if not args.cpu and not torch.cuda.is_available():
        logging.warn('CUDA is not available, proceeding with CPU') 
        args.cpu = True

    try:
        start = time.time()
        run(args)
        end = time.time()
        logging.info(f'Total running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        logging.warn('Graceful Shutdown')
    except RuntimeError:
        raise
    finally:
        if not args.cpu:    
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            logging.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')


if __name__ == '__main__':
    main()
