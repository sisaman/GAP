import torch
from typing import Annotated, Literal, Optional, Union
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from pysrc.methods.base import MethodBase
from pysrc.classifiers import MLPClassifier
from pysrc.classifiers.base import Metrics, Stage


class MLP (MethodBase):
    """non-private MLP method"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hidden_dim:      Annotated[int,   dict(help='dimension of the hidden layers')] = 16,
                 num_layers:      Annotated[int,   dict(help='number of MLP layers')] = 2,
                 activation:      Annotated[str,   dict(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, dict(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  dict(help='if true, then model uses batch normalization')] = True,
                 optimizer:       Annotated[str,   dict(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, dict(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, dict(help='weight decay (L2 penalty)')] = 0.0,
                 epochs:          Annotated[int,   dict(help='number of epochs for training')] = 100,
                 batch_size:      Annotated[Union[Literal['full'], int],   
                                                   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 **kwargs:        Annotated[dict,  dict(help='extra options passed to base class', bases=[MethodBase])]
                 ):

        assert num_layers > 1, 'number of layers must be greater than 1'
        super().__init__(num_classes, **kwargs)

        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        activation_fn = self.supported_activations[activation]

        self.classifier = MLPClassifier(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.classifier.reset_parameters()

    def fit(self, data: Data) -> Metrics:
        self.data = data
        metrics = self.train_classifier(self.data)
        metrics.update(self.test(self.data))
        return metrics

    def test(self, data: Optional[Data] = None) -> Metrics:
        if data is None:
            data = self.data
        
        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, 'test'),
            load_best=True,
        )
        return test_metics

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is None:
            data = self.data
        return self.classifier.predict(data.x)

    def train_classifier(self, data: Data) -> Metrics:
        self.classifier.to(self.device)
        self.trainer.reset()

        metrics = self.trainer.fit(
            model=self.classifier,
            epochs=self.epochs,
            optimizer=self.configure_optimizer(), 
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True
        )

        return metrics

    def data_loader(self, data: Data, stage: Stage) -> DataLoader:
        mask = data[f'{stage}_mask']
        x = data.x[mask]
        y = data.y[mask]
        if self.batch_size == 'full' or (stage != 'train' and self.full_batch_eval):
            return [(x, y)]
        else:
            return DataLoader(
                dataset=TensorDataset(x, y),
                batch_size=self.batch_size, 
                shuffle=True
            )

    def configure_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
