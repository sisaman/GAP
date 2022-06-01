import torch
import logging
from typing import Annotated, Literal, Union
import torch.nn.functional as F
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from pysrc.console import console
from pysrc.methods.base import MethodBase
from pysrc.trainer import Trainer
from pysrc.classifiers import MultiMLPClassifier
from pysrc.classifiers.base import ClassifierBase, Metrics, Stage


class GAPINF (MethodBase):
    """non-private GAP method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hops:            Annotated[int,   dict(help='number of hops', option='-k')] = 2,
                 hidden_dim:      Annotated[int,   dict(help='dimension of the hidden layers')] = 16,
                 encoder_layers:  Annotated[int,   dict(help='number of encoder MLP layers')] = 2,
                 base_layers:     Annotated[int,   dict(help='number of base MLP layers')] = 1,
                 head_layers:     Annotated[int,   dict(help='number of head MLP layers')] = 1,
                 combine:         Annotated[str,   dict(help='combination type of transformed hops', choices=MultiMLPClassifier.supported_combinations)] = 'cat',
                 activation:      Annotated[str,   dict(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, dict(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  dict(help='if true, then model uses batch normalization')] = True,
                 optimizer:       Annotated[str,   dict(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, dict(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, dict(help='weight decay (L2 penalty)')] = 0.0,
                 device:          Annotated[str,   dict(help='device to use', choices=['cpu', 'cuda'])] = 'cuda',
                 encoder_epochs:  Annotated[int,   dict(help='number of epochs for encoder pre-training (ignored if encoder_layers=0)')] = 100,
                 epochs:          Annotated[int,   dict(help='number of epochs for classifier training')] = 100,
                 batch_size:      Annotated[Union[Literal['full'], int],   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 use_amp:         Annotated[bool,  dict(help='use automatic mixed precision training')] = False,
                 ):

        super().__init__(num_classes)

        if encoder_layers == 0 and encoder_epochs > 0:
            logging.warn('encoder_layers is 0, setting encoder_epochs to 0')
            encoder_epochs = 0

        self.hops = hops
        self.encoder_layers = encoder_layers
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.encoder_epochs = encoder_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        self.use_amp = use_amp
        activation_fn = self.supported_activations[activation]

        self.encoder = MultiMLPClassifier(
            num_inputs=1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            base_layers=encoder_layers,
            head_layers=1,
            combination='cat',
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.classifier = MultiMLPClassifier(
            num_inputs=hops+1,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.trainer = Trainer(
            use_amp=self.use_amp, 
            monitor='val/acc', monitor_mode='max', 
            device=self.device,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()
        self.trainer.reset()

    def fit(self, data: Data) -> Metrics:
        self.data = data

        logging.info('step 1: encoder module')
        self.pretrain_encoder()

        logging.info('step 2: aggregation module')
        self.precompute_aggregations()

        logging.info('step 3: classification module')
        metrics = self.train_classifier()
        return metrics

    def aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        return matmul(adj_t, x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def pretrain_encoder(self):
        if self.encoder_layers > 0:
            self.encoder.to(self.device)
            self.data.x = torch.stack([self.data.x], dim=-1)

            self.trainer.reset()
            self.trainer.fit(
                model=self.encoder,
                epochs=self.encoder_epochs,
                optimizer=self.configure_optimizers(self.encoder), 
                train_dataloader=self.data_loader('train'), 
                val_dataloader=self.data_loader('val'),
                test_dataloader=None,
                checkpoint=True
            )

            self.encoder = self.trainer.load_best_model()
            self.data.x = self.encoder.encode(self.data.x)
            self.encoder.to('cpu')

    def precompute_aggregations(self):
        with console.status('computing aggregations'):
            x = F.normalize(self.data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self.aggregate(x, self.data.adj_t)
                x = self.normalize(x)
                x_list.append(x)

            self.data.x = torch.stack(x_list, dim=-1)
        
        self.data.to('cpu', 'adj_t')

    def train_classifier(self) -> Metrics:
        self.classifier.to(self.device)

        self.trainer.reset()
        metrics = self.trainer.fit(
            model=self.classifier, 
            epochs=self.epochs,
            optimizer=self.configure_optimizers(self.classifier),
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=self.data_loader('test'),
            checkpoint=False,
        )

        self.classifier.to('cpu')
        return metrics

    def data_loader(self, stage: Stage) -> DataLoader:
        mask = self.data[f'{stage}_mask']
        x = self.data.x[mask]
        y = self.data.y[mask]
        if self.batch_size == 'full' or (stage != 'train' and self.full_batch_eval):
            return [(x, y)]
        else:
            return DataLoader(
                dataset=TensorDataset(x, y),
                batch_size=self.batch_size, 
                shuffle=True
            )

    def configure_optimizers(self, model: ClassifierBase) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
