import sys
import warnings
import logging
import coloredlogs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import tabulate
from functools import partial
from args import print_args, str2bool
from datasets import Dataset, AddKNNGraph
from loggers import Logger
from models import PrivateGNN, PrivateNodeClassifier
from loader import RandomSubGraphSampler
from trainer import Trainer
from privacy import TopMFilter, GraphPerturbationEngine
from utils import timeit, colored_text, seed_everything, confidence_interval
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree

@timeit
def run(args):
    data = Dataset.from_args(args).load()
    num_classes = data.y.max().item() + 1

    stat = {
        'name': args.dataset,
        'nodes': data.num_nodes,
        'edges': data.num_edges,
        'features': data.num_features,
        'classes': int(data.y.max().item() + 1),
        'mean degree': f'{degree(data.edge_index[1], num_nodes=data.num_nodes).mean().item():.2f}',
        'median degree': degree(data.edge_index[1], num_nodes=data.num_nodes).median().item(),
        'baseline': f'{(data.y.unique(return_counts=True)[1].max().item() * 100 / data.num_nodes):.2f} %'
    }

    logging.info('dataset stats\n\n' + tabulate.tabulate([stat], headers='keys') + '\n')

    test_acc = []
    run_metrics = {}
    logger = Logger.from_args(args, enabled=args.debug, config=args)

    model = PrivateNodeClassifier.from_args(args, 
        num_features=args.hidden_dim if args.pre_train else data.num_features, 
        num_classes=num_classes, 
        inductive=args.sampling_rate<1.0,
    )

    transforms = []

    if args.add_knn:
        transforms.append(AddKNNGraph(args.add_knn))

    if args.perturbation == 'graph':
        mechanism = TopMFilter(noise_scale=args.noise_scale)
        priv_engine = GraphPerturbationEngine(mechanism, epochs=args.epochs, sampling_rate=args.sampling_rate)
        if args.noise_scale == 0.0:
            priv_engine.calibrate(eps=args.epsilon, delta=args.delta)
            transforms.append(mechanism.perturb)
    else:
        priv_engine = model.gnn.get_privacy_engine(epochs=args.epochs, sampling_rate=args.sampling_rate)
        if args.noise_scale == 0.0:
            priv_engine.calibrate(eps=args.epsilon, delta=args.delta)

    trainer = Trainer.from_args(args, 
        privacy_accountant=partial(priv_engine.get_privacy_spent, delta=args.delta), 
        device=('cpu' if args.cpu else 'cuda'),
    )
    
    for iteration in range(args.repeats):
        logging.info(f'run: {iteration + 1}')
        model.reset_parameters()
        trainer.reset()
        pre_transforms = []

        if args.pre_train:
            logger.disable()
            pt_model = PrivateNodeClassifier.from_args(args, 
                num_features=data.num_features, 
                num_classes=num_classes, 
                inductive=args.sampling_rate<1.0,
                pre_layers=1, mp_layers=0, post_layers=1
            )
            pt_trainer = Trainer.from_args(args, privacy_accountant=None, device=('cpu' if args.cpu else 'cuda'))
            pt_dataloder = RandomSubGraphSampler.from_args(args, data=data, pin_memory=not args.cpu, use_edge_sampling=False)
            pt_trainer.fit(pt_model, pt_dataloder)
            pre_transforms.append(pt_model.cpu().embed)
            logger.enabled = args.debug

        dataloader = RandomSubGraphSampler.from_args(args, 
            data=data, pin_memory=not args.cpu,
            use_edge_sampling=args.perturbation!='feature',
            transform=Compose(pre_transforms + transforms),
        )
        
        best_metrics = trainer.fit(model, dataloader)

        # process results
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
    coloredlogs.DEFAULT_FIELD_STYLES['levelname']['color'] = 'magenta'
    coloredlogs.install(level='INFO', fmt='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
    
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    # dataset args
    group_dataset = init_parser.add_argument_group('dataset arguments')
    Dataset.add_args(group_dataset)
    group_dataset.add_argument('--add-knn', '--add_knn', type=int, default=0, help='augment graph by adding k-nn edges')

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
