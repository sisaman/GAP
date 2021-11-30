from console import console
with console.status('importing modules...'):
    import logging
    import sys
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import numpy as np
    import torch
    from args import print_args, str2bool
    from datasets import Dataset, AddKNNGraph
    from loggers import Logger
    from models import PrivateNodeClassifier
    from trainer import Trainer
    from utils import timeit, seed_everything, confidence_interval
    from torch_geometric.transforms import Compose
    from torch_geometric.data import Data


@timeit
def run(args):
    with console.status('loading dataset...'):
        data_initial = Dataset.from_args(args).load(verbose=True)

    num_classes = data_initial.y.max().item() + 1
    device = 'cpu' if args.cpu else 'cuda'

    test_acc = []
    run_metrics = {}
    logger = Logger.from_args(args, enabled=args.debug, config=args)

    ### initiallize model ###

    model: PrivateNodeClassifier = PrivateNodeClassifier.from_args(args, 
        num_classes=num_classes, 
        inductive=args.sampling_rate<1.0,
    )

    ### calibrate noise to privacy budget ###
    with console.status('calibrating noise to privacy budget...'):
        model.calibrate(epsilon=args.epsilon, delta=args.delta, epochs=args.epochs, sampling_rate=args.sampling_rate)

    if args.debug:
        model.init_privacy_accountant(delta=args.delta, sampling_rate=args.sampling_rate)
    
    ### run experiment ###

    for iteration in range(args.repeats):
        data = Data(**data_initial.to_dict())
        trainer: Trainer = Trainer.from_args(args, device=device)
        model.reset_parameters()

        if args.sampling_rate == 1.0 and not args.cpu:
            with console.status('moving data to gpu...'):
                data = data.to(device)

        ### pre-training ###

        if args.pre_train:
            model.set_model_state(pre_train=True)
            trainer.fit(model, data, description='pre-training')

        ##### TODO: this should be changed #####
        # if args.sampling_rate < 1.0:
        #     transforms.append(RandomSubGraphSampler(args.sampling_rate, edge_sampling=True))

        # transforms = []

        # if args.add_knn:
        #     transforms.append(AddKNNGraph(args.add_knn))

        # data = Compose(transforms)(data)

        ### model training ###

        trainer.reset()
        model.set_model_state(pre_train=False)
        best_metrics = trainer.fit(model, data, description='training    ')

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
    group_dataset.add_argument('--add-knn', '--add_knn', type=int, default=0, help='augment graph by adding k-nn edges')
    group_dataset.add_argument('--sampling-rate', '--sampling_rate', type=float, default=1.0, help='subgraph sampling rate')

    # privacy args
    group_privacy = init_parser.add_argument_group('privacy arguments')
    group_privacy.add_argument('-e', '--epsilon', type=float, default=np.inf, help='DP epsilon parameter')
    group_privacy.add_argument('-d', '--delta', type=float, default=1e-6, help='DP delta parameter')

    # model args
    group_model = init_parser.add_argument_group('model arguments')
    PrivateNodeClassifier.add_args(group_model)

    # trainer arguments
    group_trainer = init_parser.add_argument_group('trainer arguments')
    group_trainer.add_argument('--cpu', help='train on CPU', type=str2bool, nargs='?', const=True, default=not torch.cuda.is_available())
    Trainer.add_args(group_trainer)

    # experiment args
    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-n', '--name', type=str, default=None, help='experiment name')
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
        logging.warn('CUDA is not available, running on CPU') 
        args.cpu = True

    try:
        run(args)
        if not args.cpu:    
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
            logging.info(f'Max GPU memory used = {gpu_mem:.2f} GB\n')
    except KeyboardInterrupt:
        print('\n')
        logging.warn('Graceful Shutdown...')


if __name__ == '__main__':
    main()
