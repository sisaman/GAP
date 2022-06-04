import torch
import logging
from typing import Annotated, Literal, Optional, Union
import torch.nn.functional as F
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from pysrc.console import console
from pysrc.methods.base import MethodBase
from pysrc.classifiers import Encoder, MultiMLPClassifier
from pysrc.classifiers.base import ClassifierBase, Metrics, Stage


class GAP (MethodBase):
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
                 encoder_epochs:  Annotated[int,   dict(help='number of epochs for encoder pre-training (ignored if encoder_layers=0)')] = 100,
                 epochs:          Annotated[int,   dict(help='number of epochs for classifier training')] = 100,
                 batch_size:      Annotated[Union[Literal['full'], int],   
                                                   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 **kwargs:        Annotated[dict,  dict(help='extra options passed to base class', bases=[MethodBase])]
                 ):

        super().__init__(num_classes, **kwargs)

        if encoder_layers == 0 and encoder_epochs > 0:
            logging.warn('encoder_layers is 0, setting encoder_epochs to 0')
            encoder_epochs = 0

        self.hops = hops
        self.encoder_layers = encoder_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.encoder_epochs = encoder_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        activation_fn = self.supported_activations[activation]

        self.encoder = Encoder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            head_layers=1,
            normalize=True,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.classifier = MultiMLPClassifier(
            num_inputs=hops+1,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combine,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def fit(self, data: Data) -> Metrics:
        self.data = data
        self.pretrain_encoder()
        self.precompute_aggregations()
        metrics = self.train_classifier()
        return metrics

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is None:
            data = self.data
        else:
            data.x = self.encoder.predict(data.x)
            self.precompute_aggregations(data)
        return self.classifier.predict(data.x)

    def aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        return matmul(adj_t, x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def pretrain_encoder(self):
        if self.encoder_layers > 0:
            logging.info('pretraining encoder module')
            self.encoder.to(self.device)

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
            self.data.x = self.encoder.predict(self.data.x)

    def precompute_aggregations(self, data: Optional[Data] = None):
        if data is None:
            data = self.data
        with console.status('computing aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self.aggregate(x, data.adj_t)
                x = self.normalize(x)
                x_list.append(x)

            data.x = torch.stack(x_list, dim=-1)
        
    def train_classifier(self) -> Metrics:
        logging.info('training classification module')
        self.classifier.to(self.device)

        self.trainer.reset()
        metrics = self.trainer.fit(
            model=self.classifier, 
            epochs=self.epochs,
            optimizer=self.configure_optimizers(self.classifier),
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=None,
            checkpoint=True,
        )

        test_metics = self.trainer.test(
            dataloader=self.data_loader('test'),
            load_best=True,
        )

        metrics.update(test_metics)
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
