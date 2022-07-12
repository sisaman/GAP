from core.console import console
with console.status('importing modules'):
    import torch
    import numpy as np
    import core.globals
    from time import time
    from typing import Annotated
    from rich import box
    from rich.table import Table
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from core.datasets import DatasetLoader
    from core.args.utils import print_args, strip_kwargs, create_arguments, remove_prefix
    from core.loggers import Logger
    from core.methods import supported_methods, MethodBase
    from core.attacks import supported_attacks, AttackBase
    from core.utils import seed_everything, confidence_interval
    from torch_geometric.data import Data


def run(seed:    Annotated[int,   dict(help='initial random seed')] = 12345,
        repeats: Annotated[int,   dict(help='number of times the experiment is repeated')] = 1,
        debug:   Annotated[bool, dict(help='enable global debug mode')] = False,
        **kwargs
    ):

    seed_everything(seed)

    if debug:
        console.info('debug mode enabled')
        core.globals.DEBUG_MODE = True
        console.log_level = console.DEBUG
        kwargs['project'] += '-ATTACK'

    with console.status('loading dataset'):
        loader_args = strip_kwargs(DatasetLoader, kwargs)
        data_initial = DatasetLoader(**loader_args).load(verbose=True)

    num_classes = data_initial.y.max().item() + 1
    config = dict(**kwargs, seed=seed, repeats=repeats)
    logger_args = strip_kwargs(Logger.setup, kwargs)
    logger = Logger.setup(enabled=False, config=config, **logger_args)

    ### initiallize method ###
    Method = supported_methods[kwargs.pop('method')]
    method_args = strip_kwargs(Method, kwargs, prefix='target_')
    method_args = remove_prefix(method_args, prefix='target_')
    method: MethodBase = Method(num_classes=num_classes, **method_args)

    ### initialize attack ###
    Attack = supported_attacks[kwargs['attack']]
    attack_args = strip_kwargs(Attack, kwargs)
    attack: AttackBase = Attack(method=method, **attack_args)

    run_metrics = {}

    ### run experiment ###
    for iteration in range(repeats):
        data = Data(**data_initial.to_dict())
        start_time = time()
        metrics = attack.execute(data)
        end_time = time()
        metrics['fit_time'] = end_time - start_time

        ### process results ###
        for metric, value in metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]
        

        ### print results ###
        table = Table(title=f'run {iteration + 1}', box=box.HORIZONTALS)
        table.add_column('metric')
        table.add_column('last', style="cyan")
        table.add_column('mean', style="cyan")

        table.add_row('target/train/acc', f'{run_metrics["target/train/acc"][-1]:.2f}', f'{np.mean(run_metrics["target/train/acc"]):.2f}')
        table.add_row('target/test/acc', f'{run_metrics["target/test/acc"][-1]:.2f}', f'{np.mean(run_metrics["target/test/acc"]):.2f}')

        table.add_row('shadow/train/acc', f'{run_metrics["shadow/train/acc"][-1]:.2f}', f'{np.mean(run_metrics["shadow/train/acc"]):.2f}')
        table.add_row('shadow/test/acc', f'{run_metrics["shadow/test/acc"][-1]:.2f}', f'{np.mean(run_metrics["shadow/test/acc"]):.2f}')

        for metric_name, metric_values in run_metrics.items():
            if metric_name.startswith('attack/test/'):
                table.add_row(metric_name, f'{metric_values[-1]:.2f}', f'{np.mean(metric_values):.2f}')

        console.info(table)
        console.print()

        # reset method's parameters for the next run
        attack.reset_parameters()

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
        attack_subparser = method_parser.add_subparsers(dest='attack', required=True, title='attack to perform')

        for attack_name, attack_class in supported_attacks.items():
            attack_parser = attack_subparser.add_parser(
                name=attack_name,
                help=attack_class.__doc__,
                formatter_class=ArgumentDefaultsHelpFormatter
            )

            # dataset args
            group_dataset = attack_parser.add_argument_group('dataset arguments')
            create_arguments(DatasetLoader, group_dataset)

            # target method args
            group_method = attack_parser.add_argument_group('method arguments')
            create_arguments(method_class, group_method, prefix='target_')

            # attack method args
            group_attack = attack_parser.add_argument_group('attack arguments')
            create_arguments(attack_class, group_attack)
            
            # experiment args
            group_expr = attack_parser.add_argument_group('experiment arguments')
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
