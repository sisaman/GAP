import sys
import warnings
import logging
import coloredlogs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
from args import print_args, str2bool
from datasets import Dataset, AddKNNGraph, RandomSubGraphSampler
from loggers import Logger
from models import PrivateGNN, PrivateNodeClassifier
from trainer import Trainer
from privacy import Calibrator, TopMFilter
from utils import timeit, seed_everything, confidence_interval
from torch_geometric.transforms import Compose
from torch_geometric.data import Data
    

@timeit
def run(args):
    data_initial = Dataset.from_args(args).load(verbose=True)
    num_classes = data_initial.y.max().item() + 1
    device = 'cpu' if args.cpu else 'cuda'

    test_acc = []
    run_metrics = {}
    logger = Logger.from_args(args, enabled=args.debug, config=args)

    ### initiallize model ###

    model = PrivateNodeClassifier.from_args(args, 
        num_classes=num_classes, 
        inductive=args.sampling_rate<1.0,
    )

    if args.pre_train:
        pt_model = PrivateNodeClassifier.from_args(args, 
            num_classes=num_classes, 
            inductive=args.sampling_rate<1.0,
            pre_layers=1, mp_layers=0, post_layers=1
        )

    ### calibrate noise to privacy budget ###

    if args.perturbation == 'graph':
        mechanism = TopMFilter(noise_scale=0.0) 
    else:
        mechanism = model.gnn

    mechanism_builder = lambda noise_scale: mechanism.build_mechanism(
        noise_scale=noise_scale, 
        epochs=args.epochs, 
        sampling_rate=args.sampling_rate
    )

    noise_scale = Calibrator(mechanism_builder).calibrate(eps=args.epsilon, delta=args.delta)
    mechanism.update(noise_scale=noise_scale)

    accountant = lambda epochs: mechanism.build_mechanism(
        noise_scale=noise_scale, 
        epochs=epochs,
        sampling_rate=args.sampling_rate
    ).get_approxDP(delta=args.delta)

    ### init trainer ###

    trainer: Trainer = Trainer.from_args(args, device=device)
    
    for iteration in range(args.repeats):
        logging.info(f'run: {iteration + 1}')
        data = Data(**data_initial.to_dict())
        model.reset_parameters()

        ### add data transforms ###

        transforms = []

        if args.pre_train:
            logger.disable()
            pt_model.reset_parameters()
            trainer.reset()
            trainer.fit(pt_model, data)
            transforms.append(pt_model.embed)
            logger.enabled = args.debug

        if args.sampling_rate:
            transforms.append(RandomSubGraphSampler(args.sampling_rate, edge_sampling=True))

        if args.perturbation == 'graph':
            transforms.append(mechanism.perturb)

        if args.add_knn:
            transforms.append(AddKNNGraph(args.add_knn))

        ### prepare data and train model ###

        data = Compose(transforms)(data)
        trainer.reset()
        trainer.privacy_accountant = accountant if args.debug else None
        best_metrics = trainer.fit(model, data)

        ### process results ###

        for metric, value in best_metrics.items():
            run_metrics[metric] = run_metrics.get(metric, []) + [value]

        test_acc.append(best_metrics['test/acc'])
        logging.info('test/acc: %.2f\t average: %.2f\n' % (test_acc[-1], np.mean(test_acc).item()))

    logger.enable()
    summary = {}
    
    for metric, values in run_metrics.items():
        summary[metric + '_mean'] = np.mean(values)
        summary[metric + '_std'] = np.std(values)
        summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=args.seed)
        logger.log_summary(summary)

    logger.finish()


def main():
    warnings.filterwarnings('ignore')
    coloredlogs.DEFAULT_FIELD_STYLES['levelname']['color'] = 32
    coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    Dataset.add_args(group_dataset)
    group_dataset.add_argument('--add-knn', '--add_knn', type=int, default=0, help='augment graph by adding k-nn edges')
    group_dataset.add_argument('--sampling-rate', '--sampling_rate', type=float, default=None, help='subgraph sampling rate')

    # privacy args
    group_privacy = init_parser.add_argument_group('privacy arguments')
    group_privacy.add_argument('-e', '--epsilon', type=float, default=np.inf, help='DP epsilon parameter')
    group_privacy.add_argument('-d', '--delta', type=float, default=1e-6, help='DP delta parameter')

    # model args
    group_model = init_parser.add_argument_group('model arguments')
    PrivateGNN.supported_perturbations.add('graph')
    PrivateNodeClassifier.add_args(group_model)

    # trainer arguments
    group_trainer = init_parser.add_argument_group('trainer arguments')
    group_trainer.add_argument('--pre-train', '--pre_train', help='pre-train an MLP and use its embeddings as input features', 
                                type=str2bool, nargs='?', const=True, default=False)
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
        logging.warning('CUDA is not available, running on CPU') 
        args.cpu = True

    try:
        run(args)
    except KeyboardInterrupt:
        logging.warn('Graceful Shutdown...')


if __name__ == '__main__':
    main()
