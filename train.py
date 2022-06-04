from pysrc.console import console
with console.status('importing modules'):
    import sys
    import torch
    import logging
    import numpy as np
    from time import time
    from typing import Annotated
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from pysrc.datasets import DatasetLoader
    from pysrc.args.utils import print_args, invoke, create_arguments
    from pysrc.loggers import Logger
    from pysrc.methods.base import MethodBase
    from pysrc.methods.gap import GAPINF, GAPEDP, GAPNDP
    from pysrc.methods.sage import SAGEINF, SAGEEDP, SAGENDP
    from pysrc.methods.mlp import MLP, MLPDP
    from pysrc.utils import seed_everything, confidence_interval
    from torch_geometric.data import Data

supported_methods = {
    'gap-inf': GAPINF,
    'gap-edp': GAPEDP,
    'gap-ndp': GAPNDP,
    'sage-inf': SAGEINF,
    'sage-edp': SAGEEDP,
    'sage-ndp': SAGENDP,
    'mlp': MLP,
    'mlp-dp': MLPDP
}

def run(seed:    Annotated[int, dict(help='initial random seed')] = 12345,
        repeats: Annotated[int, dict(help='number of times the experiment is repeated')] = 1,
        **kwargs
    ):

    seed_everything(seed)

    with console.status('loading dataset'):
        data_initial = invoke(DatasetLoader, **kwargs).load(verbose=True)

    test_acc = []
    run_metrics = {}
    num_classes = data_initial.y.max().item() + 1
    config = dict(**kwargs, seed=seed, repeats=repeats)
    logger = invoke(Logger.setup, enabled=False, config=config, **kwargs)

    ### initiallize method ###
    Method = supported_methods[kwargs['method']]
    method: MethodBase = invoke(Method, num_classes=num_classes, **kwargs)

    ### run experiment ###
    for iteration in range(repeats):
        data = Data(**data_initial.to_dict())
        with console.status(f'moving data to {kwargs["device"]}'):
            data.to(kwargs['device'])

        start_time = time()
        method.reset_parameters()
        metrics = method.fit(data)
        end_time = time()
        metrics['fit_time'] = end_time - start_time
        test_acc.append(metrics['test/acc'])

        ### process results ###
        for metric, value in metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]
        
        console.print()
        logging.info(f'run: {iteration + 1}/{repeats}')
        logging.info(f'test/acc: {test_acc[-1]:.2f}\t average: {np.mean(test_acc):.2f}')
        console.print()

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
    method_subparser = init_parser.add_subparsers(dest='method', required=True, title='algorithm to use')

    for method_name, method_class in supported_methods.items():
        method_parser = method_subparser.add_parser(
            name=method_name, 
            help=method_class.__doc__, 
            formatter_class=ArgumentDefaultsHelpFormatter
        )

        # dataset args
        group_dataset = method_parser.add_argument_group('dataset arguments')
        create_arguments(DatasetLoader, group_dataset)

        # method args
        group_method = method_parser.add_argument_group('method arguments')
        create_arguments(method_class, group_method)
        
        # experiment args
        group_expr = method_parser.add_argument_group('experiment arguments')
        create_arguments(run, group_expr)
        create_arguments(Logger.setup, group_expr)

    parser = ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)
    args = vars(parser.parse_args())

    print_args(args, num_cols=4)
    args['cmd'] = ' '.join(sys.argv)  # store calling command

    if args['device'] == 'cuda' and not torch.cuda.is_available():
        logging.warn('CUDA is not available, proceeding with CPU') 
        args['device'] = 'cpu'

    try:
        start = time()
        invoke(run, **args)
        end = time()
        logging.info(f'Total running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        logging.warn('Graceful Shutdown')
    except RuntimeError:
        raise
    finally:
        if args['device'] == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            logging.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')


if __name__ == '__main__':
    main()
