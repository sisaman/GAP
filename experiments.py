import yaml
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from core import console
from core.jobutils.registry import WandBJobRegistry
from core.jobutils.scheduler import JobScheduler


def create_train_commands(registry: WandBJobRegistry) -> list[str]:
    # ### Hyper-parameters
    datasets = ['facebook', 'reddit', 'amazon']
    batch_size = {'facebook': 256, 'reddit': 2048, 'amazon': 4096}

    gap_methods  = ['gap-inf', 'gap-edp', 'gap-ndp']
    sage_methods = ['sage-inf', 'sage-edp', 'sage-ndp']
    mlp_methods  = ['mlp', 'mlp-dp']
    inf_methods  = ['gap-inf', 'sage-inf']
    edp_methods  = ['gap-edp', 'sage-edp', 'mlp']
    ndp_methods  = ['gap-ndp', 'sage-ndp', 'mlp-dp']
    all_methods  = inf_methods + edp_methods + ndp_methods
    hparams = {dataset: {method: {} for method in all_methods} for dataset in datasets}

    for dataset in datasets:
        # For GAP methods
        for method in gap_methods:
            hparams[dataset][method]['encoder_layers'] = 2
            hparams[dataset][method]['base_layers'] = 1
            hparams[dataset][method]['head_layers'] = 1
            hparams[dataset][method]['combine'] = 'cat'
            hparams[dataset][method]['hops'] = [1, 2, 3, 4, 5]
        # For SAGE methods
        for method in sage_methods:
            hparams[dataset][method]['base_layers'] = 2
            hparams[dataset][method]['head_layers'] = 1
            if method != 'sage-ndp':
                hparams[dataset][method]['mp_layers'] = [1, 2, 3, 4, 5]
        # For MLP methods
        for method in mlp_methods:
            hparams[dataset][method]['num_layers'] = 3
        # For GAP-NDP and SAGE-NDP
        for method in ['gap-ndp', 'sage-ndp']:
            hparams[dataset][method]['max_degree'] = [100, 200, 300, 400]
        # For all methods
        for method in all_methods:
            hparams[dataset][method]['hidden_dim'] = 16
            hparams[dataset][method]['activation'] = 'selu'
            hparams[dataset][method]['optimizer'] = 'adam'
            hparams[dataset][method]['learning_rate'] = 0.01
            hparams[dataset][method]['repeats'] = 10
            if method in ndp_methods:
                hparams[dataset][method]['max_grad_norm'] = 1
                hparams[dataset][method]['epochs'] = 10
                hparams[dataset][method]['batch_size'] = batch_size[dataset]
            else:
                hparams[dataset][method]['batch_norm'] = True
                hparams[dataset][method]['epochs'] = 100
                hparams[dataset][method]['batch_size'] = 'full'
        # For GAP methods
        for method in gap_methods:
            hparams[dataset][method]['encoder_epochs'] = hparams[dataset][method]['epochs']

    # ### Accuracy/Privacy Trade-off
    for dataset in datasets:
        for method in all_methods:
            params = {}
            if method in ndp_methods:
                params['epsilon'] = [1, 2, 4, 8, 16]
            elif method in ['gap-edp', 'sage-edp']:
                params['epsilon'] = [0.1, 0.2, 0.5, 1, 2, 4, 8]
                
            registry.register(
                'train.py',
                method, 
                dataset=dataset,
                **params, 
                **hparams[dataset][method]
            )

    # ### Effect of Encoder
    for dataset in datasets:
        for method in ['gap-edp', 'gap-ndp']:
            hp = {**hparams[dataset][method]}
            default_encoder_layers = hp.pop('encoder_layers')
            epsilon = [0.5, 1, 2, 4, 8] if method == 'gap-edp' else [1, 2, 4, 8, 16]
            registry.register(
                'train.py',
                method,
                dataset=dataset,
                encoder_layers=[0, default_encoder_layers],
                epsilon=epsilon,
                **hp
            )

    # ### Effect of Hops
    for dataset in datasets:
        for method in ['gap-edp', 'gap-ndp']:
            hp = {**hparams[dataset][method]}
            hp.pop('hops')
            hops = [1,2,3,4,5]
            epsilon = [1, 2, 4, 8] if method == 'gap-edp' else [2, 4, 8, 16]
            registry.register(
                'train.py',
                method,
                dataset=dataset,
                hops=hops,
                epsilon=epsilon,
                **hp
            )

    # ### Effect of Degree
    for dataset in datasets:
        method = 'gap-ndp'
        hp = {**hparams[dataset][method]}
        hp.pop('max_degree')
        max_degree = [10,20,50,100,200,300,400]
        epsilon = [2, 4, 8, 16]
        registry.register(
            'train.py',
            method,
            dataset=dataset,
            max_degree=max_degree,
            epsilon=epsilon,
            **hp
        )

    return registry.job_list
    

def create_attack_commands(registry: WandBJobRegistry) -> list[str]:
    # Hyperparameters
    datasets = ['facebook', 'reddit', 'amazon']
    gap_methods  = ['gap-inf', 'gap-ndp']
    sage_methods = ['sage-inf', 'sage-ndp']
    mlp_methods  = ['mlp', 'mlp-dp']
    ndp_methods  = ['gap-ndp', 'sage-ndp', 'mlp-dp']
    all_methods  = gap_methods + sage_methods + mlp_methods
    hparams = {dataset: {method: {} for method in all_methods} for dataset in datasets}

    for dataset in datasets:
        # For GAP methods
        for method in gap_methods:
            hparams[dataset][method]['shadow_encoder_layers'] = 2
            hparams[dataset][method]['shadow_base_layers'] = 1
            hparams[dataset][method]['shadow_head_layers'] = 1
            hparams[dataset][method]['shadow_combine'] = 'cat'
            hparams[dataset][method]['shadow_hops'] = 2
        # For SAGE methods
        for method in sage_methods:
            hparams[dataset][method]['shadow_base_layers'] = 2
            hparams[dataset][method]['shadow_head_layers'] = 1
            if method != 'sage-ndp':
                hparams[dataset][method]['shadow_mp_layers'] = 2
        # For MLP methods
        for method in mlp_methods:
            hparams[dataset][method]['shadow_num_layers'] = 3
        # For GAP-NDP and SAGE-NDP
        for method in ['gap-ndp', 'sage-ndp']:
            hparams[dataset][method]['shadow_max_degree'] = 100
        # For all methods
        for method in all_methods:
            hparams[dataset][method]['shadow_hidden_dim'] = 64
            hparams[dataset][method]['shadow_activation'] = 'selu'
            hparams[dataset][method]['shadow_optimizer'] = 'adam'
            hparams[dataset][method]['shadow_learning_rate'] = 0.01
            if method in ndp_methods:
                hparams[dataset][method]['shadow_max_grad_norm'] = 1
                hparams[dataset][method]['shadow_epochs'] = 10
                hparams[dataset][method]['shadow_batch_size'] = 256
            else:
                hparams[dataset][method]['shadow_batch_norm'] = True
                hparams[dataset][method]['shadow_epochs'] = 100
                hparams[dataset][method]['shadow_batch_size'] = 'full'
            if method != 'sage-ndp':
                hparams[dataset][method]['shadow_val_interval'] = 0
            if method in gap_methods:
                hparams[dataset][method]['shadow_encoder_epochs'] = hparams[dataset][method]['shadow_epochs']
            hparams[dataset][method]['num_nodes_per_class'] = 1000
            hparams[dataset][method]['attack_hidden_dim'] = 64
            hparams[dataset][method]['attack_num_layers'] = 3
            hparams[dataset][method]['attack_activation'] = 'selu'
            hparams[dataset][method]['attack_batch_norm'] = True
            hparams[dataset][method]['attack_batch_size'] = 'full'
            hparams[dataset][method]['attack_epochs'] = 100
            hparams[dataset][method]['attack_optimizer'] = 'adam'
            hparams[dataset][method]['attack_learning_rate'] = 0.01
            hparams[dataset][method]['attack_val_interval'] = 1
            hparams[dataset][method]['repeats'] = 10

    for dataset in datasets:
        for method in all_methods:
            params = {}
            if method in ndp_methods:
                params['shadow_epsilon'] = [1, 2, 4, 8, 16]
            
            registry.register(
                'attack.py',
                method, 
                'nmi',
                dataset=dataset,
                **params, 
                **hparams[dataset][method]
            )

    return registry.job_list


def generate(path: str):
    """Generate experiment job file.

    Args:
        path (str): Path to store job file.
    """
    with open('wandb.yaml') as f:
        wandb_config = yaml.safe_load(f)

    registry_train = WandBJobRegistry(
        entity=wandb_config['username'], 
        project=wandb_config['project']['train']
    )

    registry_attack = WandBJobRegistry(
        entity=wandb_config['username'], 
        project=wandb_config['project']['attack']
    )

    with console.status('pulling jobs from WandB'):
        registry_train.pull()
        registry_attack.pull()

    with console.status('generating job commands'):
        train_commands = create_train_commands(registry_train)
        attack_commands = create_attack_commands(registry_attack)

    job_list = train_commands + attack_commands
    console.info(f'{len(job_list)} jobs generated')
    with console.status(f'saving jobs to {path}'):
        registry_train.job_list = job_list
        registry_train.save(path=path)


def run(job_file: str, scheduler_name: str) -> None:
    """Run jobs in parallel using a distributed job scheduler.

    Args:
        job_file (str): Path to the job file.
        scheduler_name (str): Name of the scheduler to use.
    """
    scheduler = JobScheduler(job_file=job_file, scheduler=scheduler_name)
    scheduler.submit()


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--generate', action='store_true', help='Generate jobs')
    parser.add_argument('--run', action='store_true', help='Run jobs')
    parser.add_argument('--path', type=str, default='jobs/gap.sh', help='Path to the job file')
    parser.add_argument('--scheduler', type=str, default='sge', help='Job scheduler to use', 
                        choices=JobScheduler.cluster_resolver.options)
    args = parser.parse_args()

    if args.generate:
        generate(args.path)
    if args.run:
        run(job_file=args.path, scheduler_name=args.scheduler)
    
    if not args.generate and not args.run:
        parser.error('Please specify either --generate or --run')


if __name__ == '__main__':
    main()
