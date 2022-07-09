from abc import ABC, abstractmethod
from typing import Annotated, Iterable, Literal, Optional
from torch import Tensor
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from core.classifiers.base import ClassifierBase, Metrics, Stage
from core.console import console
from core.trainer import Trainer


class MethodBase(ABC):

    @abstractmethod
    def fit(self, data: Data, prefix: str = '') -> Metrics:
        """Fit the model to the given data."""

    @abstractmethod
    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        """Test the model on the given data, or the training data if data is None."""

    @abstractmethod
    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""


class NodeClassificationBase(MethodBase):
    def __init__(self, 
                 num_classes:     int, 
                 epochs:          Annotated[int,   dict(help='number of epochs for training')] = 100,
                 optimizer:       Annotated[str,   dict(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, dict(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, dict(help='weight decay (L2 penalty)')] = 0.0,
                 val_interval:    Annotated[int, dict(help='interval of validation')] = 1,
                 use_amp:         bool = False,
                 device:          Literal['cpu', 'cuda'] = 'cuda', 
                 ):

        self.num_classes = num_classes
        self.device = device
        self.use_amp = use_amp
        self.val_interval = val_interval
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.trainer = Trainer(
            val_interval=val_interval,
            use_amp=self.use_amp, 
            monitor='val/acc', 
            monitor_mode='max', 
            device=self.device,
        )
        self.data = None  # data is kept for caching purposes

    @property
    @abstractmethod
    def classifier(self) -> ClassifierBase:
        """Return the underlying classifier."""

    def reset_parameters(self):
        self.classifier.reset_parameters()
        self.trainer.reset()
        self.data = None

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        self.data = data
        train_metrics = self._train(self.data, prefix=prefix)
        test_metrics = self.test(self.data, prefix=prefix)
        return {**train_metrics, **test_metrics}

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if data is None:
            data = self.data
        
        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, 'test'),
            prefix=prefix,
        )
        return test_metics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        if data is None:
            data = self.data
        return self.classifier.predict(data)

    def _train(self, data: Data, prefix: str = '') -> Metrics:
        console.info('training classifier')
        self.classifier.to(self.device)

        metrics = self.trainer.fit(
            model=self.classifier,
            epochs=self.epochs,
            optimizer=self._configure_optimizer(),
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=None,
            checkpoint=True,
            prefix=prefix,
        )

        return metrics

    def _configure_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    @abstractmethod
    def data_loader(self, data: Data, stage: Stage) -> Iterable:
        """Return a dataloader for the given stage."""
