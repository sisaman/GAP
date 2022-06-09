from core.args.utils import remove_prefix
from core.console import console
with console.status('importing modules'):
    import torch
    import numpy as np
    from time import time
    from typing import Annotated
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from core.datasets import DatasetLoader
    from core.args.utils import print_args, strip_kwargs, create_arguments, remove_prefix
    from core.loggers import Logger
    from core.methods.base import MethodBase
    from core.methods.gap import GAP, EdgePrivGAP, NodePrivGAP
    from core.methods.sage import SAGE, EdgePrivSAGE, NodePrivSAGE
    from core.methods.mlp import MLP, PrivMLP
    from core.attacks import AttackBase, LinkStealingAttack, NodeMembershipInference
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

supported_attacks = {
    'lsa': LinkStealingAttack,
    'nmi': NodeMembershipInference,
}

def run(device:  Annotated[str,   dict(help='device to use', choices=['cpu', 'cuda'])] = 'cuda',
        use_amp: Annotated[bool,  dict(help='use automatic mixed precision training')] = False,
        seed:    Annotated[int,   dict(help='initial random seed')] = 12345,
        repeats: Annotated[int,   dict(help='number of times the experiment is repeated')] = 1,
        **kwargs
    ):

    seed_everything(seed)

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
    method: MethodBase = Method(
        num_classes=num_classes, 
        device=device,
        use_amp=use_amp,
        **method_args
    )

    ### initialize attack ###
    Attack = supported_attacks[kwargs['attack']]
    attack_args = strip_kwargs(Attack, kwargs)
    attack: AttackBase = Attack(
        method=method, 
        device=device,
        use_amp=use_amp,
        **attack_args
    )

    run_metrics = {}

    ### run experiment ###
    for iteration in range(repeats):
        data = Data(**data_initial.to_dict())
        with console.status(f'moving data to {device}'):
            data.to(device)

        start_time = time()
        metrics = attack.execute(data)
        end_time = time()
        metrics['fit_time'] = end_time - start_time

        ### process results ###
        for metric, value in metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]
        
        console.print()
        console.info(f'run: {iteration + 1}/{repeats}')
        console.info(f'target/train/acc: {run_metrics["target/train/acc"][-1]:.2f}\t average: {np.mean(run_metrics["target/train/acc"]):.2f}')
        console.info(f'target/test/acc: {run_metrics["target/test/acc"][-1]:.2f}\t average: {np.mean(run_metrics["target/test/acc"]):.2f}')
        console.info(f'shadow/train/acc: {run_metrics["shadow/train/acc"][-1]:.2f}\t average: {np.mean(run_metrics["shadow/train/acc"]):.2f}')
        console.info(f'shadow/test/acc: {run_metrics["shadow/test/acc"][-1]:.2f}\t average: {np.mean(run_metrics["shadow/test/acc"]):.2f}')
        console.info(f'attack/test/acc: {run_metrics["attack/test/acc"][-1]:.2f}\t average: {np.mean(run_metrics["attack/test/acc"]):.2f}')
        console.info(f'attack/adv: {run_metrics["attack/adv"][-1]:.4f}\t average: {np.mean(run_metrics["attack/adv"]):.4f}')
        console.print()

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

    if kwargs['device'] == 'cuda' and not torch.cuda.is_available():
        console.warning('CUDA is not available, proceeding with CPU') 
        kwargs['device'] = 'cpu'

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
        if kwargs['device'] == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            console.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')


if __name__ == '__main__':
    main()
