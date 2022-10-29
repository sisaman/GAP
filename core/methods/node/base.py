from abc import abstractmethod
from typing import Annotated, Literal, Optional, Union
import torch
from torch import Tensor
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from core.data.loader import NodeDataLoader
from core import globals
from core.methods.base import MethodBase
from core.modules.base import TrainableModule, Metrics, Stage
from core import console
from core.trainer import Trainer


class NodeClassification(MethodBase):
    def __init__(self, 
                 num_classes:     int, 
                 epochs:          Annotated[int,   dict(help='number of epochs for training')] = 100,
                 optimizer:       Annotated[str,   dict(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, dict(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, dict(help='weight decay (L2 penalty)')] = 0.0,
                 batch_size:      Annotated[Union[Literal['full'], int],   
                                                   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 device:          Annotated[str,   dict(help='device to use', choices=['cpu', 'cuda'])] = 'cuda',
                 **trainer_args:  Annotated[dict,  dict(help='extra options passed to the trainer class', bases=[Trainer])]
                 ):

        self.num_classes = num_classes
        self.epochs = epochs
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        self.device = device

        if self.device == 'cuda' and not torch.cuda.is_available():
            console.warning('CUDA is not available, proceeding with CPU') 
            self.device = 'cpu'

        self.data = None  # data is kept for caching purposes
        self.trainer = self.configure_trainer(**trainer_args)

    @property
    @abstractmethod
    def classifier(self) -> TrainableModule:
        """Return the underlying classifier."""

    def reset_parameters(self):
        self.classifier.reset_parameters()
        self.trainer.reset()
        self.data = None

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        """Fit the model to the given data."""
        self.data = data.to(self.device, non_blocking=True)
        train_metrics = self._train(self.data, prefix=prefix)
        test_metrics = self.test(self.data, prefix=prefix)
        return {**train_metrics, **test_metrics}

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        if data is None:
            data = self.data
        
        data = data.to(self.device, non_blocking=True)

        test_metics = self.trainer.test(
            dataloader=self.data_loader(data, 'test'),
            prefix=prefix,
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        if data is None:
            data = self.data

        data = data.to(self.device, non_blocking=True)
        return self.classifier.predict(data)

    def _train(self, data: Data, prefix: str = '') -> Metrics:
        console.info('training classifier')
        self.classifier.to(self.device)

        metrics = self.trainer.fit(
            model=self.classifier,
            epochs=self.epochs,
            optimizer=self.configure_optimizer(),
            train_dataloader=self.data_loader(data, 'train'), 
            val_dataloader=self.data_loader(data, 'val'),
            test_dataloader=self.data_loader(data, 'test') if globals['debug'] else None,
            checkpoint=True,
            prefix=prefix,
        )

        return metrics

    def configure_trainer(self, **kwargs) -> Trainer:
        trainer = Trainer(
            monitor='val/acc', 
            monitor_mode='max', 
            **kwargs
        )
        return trainer

    def configure_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def data_loader(self, data: Data, stage: Stage) -> NodeDataLoader:
        """Return a dataloader for the given stage."""
        
        batch_size = 'full' if (stage != 'train' and self.full_batch_eval) else self.batch_size
        dataloader = NodeDataLoader(
            data=data, 
            stage=stage,
            batch_size=batch_size, 
            shuffle=(stage == 'train'), 
            drop_last=True,
            poisson_sampling=False,
        )

        return dataloader
