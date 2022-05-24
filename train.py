from pysrc.console import console
with console.status('importing modules'):
    import sys
    import time
    import torch
    import logging
    import numpy as np
    from typing import Annotated
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from pysrc.datasets import DatasetLoader
    from pysrc.args.utils import print_args, invoke_by_args, create_arguments
    from pysrc.loggers import Logger
    from pysrc.methods import GAP, GraphSAGE
    from pysrc.utils import seed_everything, confidence_interval
    from torch_geometric.data import Data


def run(
    name: Annotated[str, dict(help='experiment name')] = '',
    seed: Annotated[int, dict(help='initial random seed')] = 12345,
    repeats: Annotated[int, dict(help='number of times the experiment is repeated')] = 1,
    **kwargs
    ):
    with console.status('loading dataset'):
        data_initial = invoke_by_args(DatasetLoader, kwargs).load(verbose=True)

    test_acc = []
    run_metrics = {}
    num_classes = data_initial.y.max().item() + 1
    config = dict(**kwargs, name=name, seed=seed, repeats=repeats)
    logger = invoke_by_args(Logger.setup, dict(enabled=False, config=config, **kwargs))

    ### initiallize model ###
    Method = GAP if kwargs['model'] == 'gap' else GraphSAGE
    model = invoke_by_args(Method, dict(num_classes=num_classes, **kwargs))

    ### run experiment ###
    for iteration in range(repeats):
        data = Data(**data_initial.to_dict())

        model.reset_parameters()
        metrics = model.fit(data)

        ### process results ###
        for metric, value in metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

        test_acc.append(metrics['test/acc'])
        console.print()
        logging.info(f'run: {iteration + 1}\t test/acc: {test_acc[-1]:.2f}\t average: {np.mean(test_acc).item():.2f}\n')

    logger.enable()
    summary = {}
    
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=seed)
        logger.log_summary(summary)

    logger.finish()
    print()


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    command_subparser = init_parser.add_subparsers(dest='model', required=True, title='model architecture')
    command_parser = {
        'gap': command_subparser.add_parser('gap', help='GAP model', formatter_class=ArgumentDefaultsHelpFormatter),
        'sage': command_subparser.add_parser('sage', help='GraphSAGE model', formatter_class=ArgumentDefaultsHelpFormatter),
    }

    for parser_name, parser in command_parser.items():
        # dataset args
        group_dataset = parser.add_argument_group('dataset arguments')
        create_arguments(DatasetLoader, group_dataset)

        # model args
        group_model = parser.add_argument_group('model arguments')
        if parser_name == 'gap':
            create_arguments(GAP, group_model)
        elif parser_name == 'sage':
            create_arguments(GraphSAGE, group_model)

        # experiment args
        group_expr = parser.add_argument_group('experiment arguments')
        create_arguments(run, group_expr)
        create_arguments(Logger.setup, group_expr)

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
        invoke_by_args(run, args)
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
