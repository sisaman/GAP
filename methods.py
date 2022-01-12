from console import console
import torch
import numpy as np
import logging
from torch.optim import Adam, SGD
from args import support_args
from privacy import TopMFilter, NoisySGD, PMA, ComposedNoisyMechanism
from torch.utils.data import DataLoader, TensorDataset
from trainer import Trainer
from datasets import NeighborSampler
from models import MultiStageClassifier, supported_activations


@support_args
class GAP:
    supported_dp_levels = {'edge', 'node'}
    supported_perturbations = {'aggr', 'graph'}

    def __init__(self,
                 num_classes,
                 dp_level:      dict(help='level of privacy protection', choices=supported_dp_levels) = 'edge',
                 epsilon:       dict(help='DP epsilon parameter', option='-e') = np.inf,
                 delta:         dict(help='DP delta parameter', option='-d') = 1e-6,
                 perturbation:  dict(help='perturbation method', option='-p', choices=supported_perturbations) = 'aggr',
                 hops:          dict(help='number of hops', option='-k') = 2,
                 max_degree:    dict(help='max degree per each node') = 0,
                 hidden_dim:    dict(help='dimension of the hidden layers') = 16,
                 encoder_layers:dict(help='number of encoder MLP layers') = 2,
                 pre_layers:    dict(help='number of pre-combination MLP layers') = 1,
                 post_layers:   dict(help='number of post-combination MLP layers') = 1,
                 combine:       dict(help='combination type of transformed hops', choices=MultiStageClassifier.supported_combinations) = 'cat',
                 activation:    dict(help='type of activation function', choices=supported_activations) = 'selu',
                 dropout:       dict(help='dropout rate (between zero and one)') = 0.0,
                 batchnorm:     dict(help='if True, then model uses batch normalization') = True,
                 optimizer:     dict(help='optimization algorithm', choices=['sgd', 'adam']) = 'adam',
                 learning_rate: dict(help='learning rate', option='--lr') = 0.01,
                 weight_decay:  dict(help='weight decay (L2 penalty)') = 0.0,
                 cpu:           dict(help='if True, then model is trained on CPU') = False,
                 pre_epochs:    dict(help='number of epochs for pre-training') = 100,
                 epochs:        dict(help='number of epochs for training') = 100,
                 batch_size:    dict(help='batch size (if zero, performs full-batch training)') = 0,
                 max_grad_norm: dict(help='maximum norm of the per-sample gradients (ignored when using edge-level DP)') = 1.0,
                 use_amp:       dict(help='use automatic mixed precision training') = False,
                 ):

        super().__init__()

        assert dp_level == 'edge' or max_degree > 0 
        assert dp_level == 'edge' or batch_size > 0
        assert encoder_layers == 0 or pre_epochs > 0

        self.dp_level = dp_level
        self.epsilon = epsilon
        self.delta = delta
        self.perturbation = perturbation
        self.hops = hops
        self.max_degree = max_degree
        self.encoder_layers = encoder_layers
        self.device = 'cpu' if cpu else 'cuda'
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.pre_epochs = pre_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.noise_scale = 1.0 # will be calibrated later

        self.encoder = MultiStageClassifier(
            num_stages=1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=encoder_layers,
            post_layers=1,
            combination_type='cat',
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        self.classifier = MultiStageClassifier(
            num_stages=hops+1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            pre_layers=pre_layers,
            post_layers=post_layers,
            combination_type=combine,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def set_training_state(self, pre_train):
        self._pre_train_flag = pre_train

    def init_privacy_mechanisms(self):
        self.pma_mechanism = PMA(noise_scale=self.noise_scale, hops=self.hops)

        if self.perturbation == 'graph':
            self.graph_mechanism = TopMFilter(noise_scale=self.noise_scale)
            mechanism_list = [self.graph_mechanism]
        elif self.dp_level == 'edge':
            mechanism_list = [self.pma_mechanism]
        elif self.dp_level == 'node':
            dataset_size = len(self.data_loader('train').dataset)

            self.pretraining_noisy_sgd = NoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=dataset_size, 
                batch_size=self.batch_size, 
                epochs=self.pre_epochs,
                max_grad_norm=self.max_grad_norm,
            )

            self.training_noisy_sgd = NoisySGD(
                noise_scale=self.noise_scale, 
                dataset_size=dataset_size, 
                batch_size=self.batch_size, 
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
            )

            mechanism_list = [self.pretraining_noisy_sgd, self.pma_mechanism, self.training_noisy_sgd]


        composed_mech = ComposedNoisyMechanism(
            noise_scale=self.noise_scale, 
            mechanism_list=mechanism_list, 
            coeff_list=[1]*len(mechanism_list)
        )

        with console.status('calibrating noise to privacy budget...'):
            noise_scale = composed_mech.calibrate(eps=self.epsilon, delta=self.delta)
            logging.info(f'noise scale: {noise_scale:.4f}\n')

    def fit(self, data):
        self.data = NeighborSampler(self.max_degree)(data)
        self.data.to(self.device, 'adj_t')
        self.init_privacy_mechanisms()
        
        if self.adj_pert:
            self.pma_mechanism.update(noise_scale=0)
            with console.status('applying adjacency matrix perturbations...'):
                self.data = self.graph_mechanism(self.data)

        with console.status(f'moving data to {self.device}...'):
            self.data.to(self.device)

        logging.info('pretraining encoder module...')
        self.pretrain_encoder()

        logging.info('precomputing aggregation module...')
        self.precompute_aggregations()

        logging.info('training classification module...')
        return self.train_classifier()

    def pretrain_encoder(self):
        if self.encoder_layers > 0:
            self.set_training_state(pre_train=True)
            self.data.x = torch.stack([self.data.x], dim=-1)

            trainer = Trainer(
                epochs=self.pre_epochs, 
                use_amp=self.use_amp, 
                monitor='val/loss', monitor_mode='min', 
                device=self.device,
                dp_mechanism=self.pretraining_noisy_sgd if self.dp_level == 'node' else None,
            )

            trainer.fit(
                model=self.encoder,
                optimizer=self.configure_optimizers(self.encoder), 
                train_dataloader=self.data_loader('train'), 
                val_dataloader=self.data_loader('val'),
                test_dataloader=None,
                checkpoint=True
            )

            self.encoder = trainer.load_best_model()
            self.data.x = self.encoder.encode(self.data.x)

    def precompute_aggregations(self):
        sensitivity = 1 if self.dp_level == 'edge' else np.sqrt(self.max_degree)
        self.data = self.pma_mechanism(self.data, sensitivity=sensitivity)

    def train_classifier(self):
        self.set_training_state(pre_train=False)

        trainer = Trainer(
            epochs=self.epochs, 
            use_amp=self.use_amp, 
            monitor='val/loss', monitor_mode='min', 
            device=self.device,
            dp_mechanism=self.training_noisy_sgd if self.dp_level == 'node' else None,
        )

        metrics = trainer.fit(
            model=self.classifier, 
            optimizer=self.configure_optimizers(self.classifier),
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=self.data_loader('test'),
            checkpoint=False,
        )

        return metrics

    def data_loader(self, stage):
        mask = self.data[f'{stage}_mask']
        x = self.data.x[mask]
        y = self.data.y[mask]

        if self.batch_size == 0:
            return TensorDataset(x.unsqueeze(0), y.unsqueeze(0))
        else:
            dataset = TensorDataset(x, y)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self, model):
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
