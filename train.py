from core.console import console
with console.status('importing modules'):
    import torch
    import numpy as np
    from time import time
    from typing import Annotated
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from core.datasets import DatasetLoader
    from core.args.utils import print_args, create_arguments, strip_kwargs
    from core.loggers import Logger
    from core.methods.base import NodeClassificationBase
    from core.methods.gap import GAP, EdgePrivGAP, NodePrivGAP
    from core.methods.sage import SAGE, EdgePrivSAGE, NodePrivSAGE
    from core.methods.mlp import MLP, PrivMLP
    from core.utils import seed_everything, confidence_interval
    from torch_geometric.data import Data

supported_methods = {
    'gap-inf':  GAP,
    'gap-edp':  EdgePrivGAP,
    'gap-ndp':  NodePrivGAP,
    'sage-inf': SAGE,
    'sage-edp': EdgePrivSAGE,
    'sage-ndp': NodePrivSAGE,
    'mlp':      MLP,
    'mlp-dp':   PrivMLP
}

def run(seed:    Annotated[int,   dict(help='initial random seed')] = 12345,
        repeats: Annotated[int,   dict(help='number of times the experiment is repeated')] = 1,
        **kwargs
    ):

    seed_everything(seed)

    with console.status('loading dataset'):
        loader_args = strip_kwargs(DatasetLoader, kwargs)
        data_initial = DatasetLoader(**loader_args).load(verbose=True)

    test_acc = []
    run_metrics = {}
    num_classes = data_initial.y.max().item() + 1
    config = dict(**kwargs, seed=seed, repeats=repeats)
    logger_args = strip_kwargs(Logger.setup, kwargs)
    logger = Logger.setup(enabled=False, config=config, **logger_args)

    ### initiallize method ###
    Method = supported_methods[kwargs['method']]
    method_args = strip_kwargs(Method, kwargs)
    method: NodeClassificationBase = Method(num_classes=num_classes, **method_args)

    ### run experiment ###
    for iteration in range(repeats):
        data = Data(**data_initial.to_dict())
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
        console.info(f'run: {iteration + 1}/{repeats}')
        console.info(f'test/acc: {test_acc[-1]:.2f}\t average: {np.mean(test_acc):.2f}')
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
    kwargs = vars(parser.parse_args())
    print_args(kwargs, num_cols=4)

    try:
        start = time()
        run(**kwargs)
        end = time()
        console.info(f'Total running time: {(end - start):.2f} seconds.')
    except KeyboardInterrupt:
        print('\n')
        console.warning('Graceful Shutdown')
    except RuntimeError:
        raise
    finally:
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            console.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')


if __name__ == '__main__':
    main()
