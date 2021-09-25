import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch

from args import Enum, EnumAction, print_args, str2bool
from datasets import Dataset
from loggers import Logger
from models import PrivateNodeClassifier
from trainer import Trainer
from privacy import GaussianMechanism, TopMFilter, LaplaceMechanism
from utils import timeit, colored_text, seed_everything, confidence_interval


class Perturbation(Enum):
    Graph = 'graph'
    Aggregation = 'aggr'
    Feature = 'feature'


class Mechanism(Enum):
    Gaussian = 'gaussian'
    Laplace = 'laplace'


@timeit
def run(args):
    dataset = Dataset.from_args(args).load()
    num_classes = dataset.y.max().item() + 1

    test_acc = []
    run_metrics = {}
    logger = Logger.from_args(args, enabled=args.debug, config=args)
    
    for iteration in range(args.repeats):
        data = dataset.clone().to('cpu' if args.cpu else 'cuda')
        model = PrivateNodeClassifier.from_args(args, input_dim=data.num_features, num_classes=num_classes)

        if args.perturbation == Perturbation.Graph:
            mechanism = TopMFilter(eps_edges=0.9*args.epsilon, eps_count=0.1*args.epsilon)
            data = mechanism.perturb(data)
        else:
            MechCls = GaussianMechanism if args.mechanism == Mechanism.Gaussian else LaplaceMechanism
            mechanism = MechCls(noise_std=args.noise_std, delta=args.delta)
            model.set_privacy_mechanism(mechanism=mechanism, perturbation_mode=args.perturbation.value)
        
        trainer = Trainer.from_args(args, privacy_accountant=mechanism.get_privacy_spent)
        best_metrics = trainer.fit(model, data)

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
    group_privacy.add_argument('-d', '--delta', type=float, default=1e-6, help='DP delta parameter')
    group_privacy.add_argument('-n', '--noise-std', type=float, default=0, help='standard deviation of the noise')
    group_dataset.add_argument('-m', '--mechanism', type=Mechanism, action=EnumAction, default=Mechanism.Gaussian, help='perturbation mechanism')

    # trainer arguments
    group_trainer = init_parser.add_argument_group('trainer arguments')
    group_trainer.add_argument('--cpu', help='train on CPU', type=str2bool, nargs='?', const=True, default=not torch.cuda.is_available())
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
