import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch

from args import Enum, EnumAction, print_args, str2bool
from datasets import Dataset
from loggers import Logger
from models import PrivateNodeClassifier
from loader import RandomSubGraphSampler
from trainer import Trainer
from privacy import NoisyMechanism, TopMFilter
from utils import timeit, colored_text, seed_everything, confidence_interval


class Perturbation(Enum):
    Graph = 'graph'
    Aggregation = 'aggr'
    Feature = 'feature'


@timeit
def run(args):
    data = Dataset.from_args(args).load()
    num_classes = data.y.max().item() + 1

    test_acc = []
    run_metrics = {}
    logger = Logger.from_args(args, enabled=args.debug, config=args)
    
    for iteration in range(args.repeats):
        # data = dataset.clone().to('cpu' if args.cpu else 'cuda')
        model = PrivateNodeClassifier.from_args(args, input_dim=data.num_features, num_classes=num_classes)

        if args.perturbation == Perturbation.Graph:
            mechanism = TopMFilter(eps_edges=0.9*args.epsilon, eps_count=0.1*args.epsilon)
            data = mechanism.perturb(data)
        else:
            mechanism = NoisyMechanism.from_args(args)
            model.set_privacy_mechanism(mechanism=mechanism, perturbation_mode=args.perturbation.value)
        
        dataloader = RandomSubGraphSampler.from_args(args, 
            data=data, pin_memory=not args.cpu,
            edge_sampler=args.perturbation==Perturbation.Feature, 
        )
        
        trainer = Trainer.from_args(args, 
            privacy_accountant=mechanism.get_privacy_spent, 
            device=('cpu' if args.cpu else 'cuda'),
        )

        best_metrics = trainer.fit(model, dataloader)

        # process results
        for metric, value in best_metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

        test_acc.append(best_metrics['test/acc'])
        print('\nrun: %d\ntest/acc: %.2f\t average: %.2f' % (iteration+1, test_acc[-1], np.mean(test_acc).item()))
        print('eps:', best_metrics['eps'], '\n')

    logger.enable()
    summary = {}
    
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=args.seed)
        logger.log_summary(summary)

    logger.finish()


def main():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    Dataset.add_args(group_dataset)

    # model args
    group_model = init_parser.add_argument_group('model arguments')
    PrivateNodeClassifier.add_args(group_model)

    # privacy args
    group_privacy = init_parser.add_argument_group('privacy arguments')
    group_privacy.add_argument('-p', '--perturbation', type=Perturbation, action=EnumAction, default=Perturbation.Feature, help='perturbation method')
    group_privacy.add_argument('-e', '--epsilon', type=float, default=np.inf, help='DP epsilon parameter')
    NoisyMechanism.add_args(group_privacy)

    # trainer arguments
    group_trainer = init_parser.add_argument_group('trainer arguments')
    group_trainer.add_argument('--cpu', help='train on CPU', type=str2bool, nargs='?', const=True, default=not torch.cuda.is_available())
    RandomSubGraphSampler.add_args(group_trainer)
    Trainer.add_args(group_trainer)

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=12345, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help="number of times the experiment is repeated")
    Logger.add_args(group_expr)

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
