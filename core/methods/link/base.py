from abc import abstractmethod
from typing import Annotated, Optional

import torch
from core import globals
from core import console
from core.args.utils import ArgInfo
from core.methods.base import MethodBase
from core.modules.base import Metrics, TrainableModule
from core.trainer import Trainer
from torch import Tensor
from torch.optim import SGD, Adam, Optimizer
from torch_geometric.data import Data


class LinkPrediction(MethodBase):
    def __init__(self, 
                 epochs:          Annotated[int,   ArgInfo(help='number of epochs for training')] = 100,
                 optimizer:       Annotated[str,   ArgInfo(help='optimization algorithm', choices=['sgd', 'adam'])] = 'adam',
                 learning_rate:   Annotated[float, ArgInfo(help='learning rate', option='--lr')] = 0.01,
                 weight_decay:    Annotated[float, ArgInfo(help='weight decay (L2 penalty)')] = 0.0,
                 device:          Annotated[str,   ArgInfo(help='device to use', choices=['cpu', 'cuda'])] = 'cuda',
                 **trainer_args:  Annotated[dict,  ArgInfo(help='extra options passed to the trainer class', bases=[Trainer])]
                 ):

        self.epochs = epochs
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device

        if self.device == 'cuda' and not torch.cuda.is_available():
            console.warning('CUDA is not available, proceeding with CPU') 
            self.device = 'cpu'

        self.trainer = self.configure_trainer(**trainer_args)

    @property
    @abstractmethod
    def link_predictor(self) -> TrainableModule:
        """Return the underlying classifier."""

    def reset_parameters(self):
        self.link_predictor.reset_parameters()
        self.trainer.reset()

    def fit(self, train_data: Data, val_data: Optional[Data] = None, test_data: Optional[Data] = None, prefix: str = '') -> Metrics:
        """Fit the model to the given data."""
        with console.status(f'moving data to {self.device}'):
            train_data = train_data.to(self.device, non_blocking=True)
            if val_data is not None:
                val_data = val_data.to(self.device, non_blocking=True)
            if test_data is not None:
                test_data = test_data.to(self.device, non_blocking=True)
        
        train_metrics = self._train(train_data, val_data, test_data, prefix=prefix)
        test_metrics = self.test(test_data, prefix=prefix) if test_data is not None else {}
        return {**train_metrics, **test_metrics}

    def test(self, data: Data, prefix: str = '') -> Metrics:
        """Predict the labels for the given data, or the training data if data is None."""
        
        with console.status(f'moving test data to {self.device}'):
            data = data.to(self.device, non_blocking=True)

        test_metics = self.trainer.test(
            dataloader=[data],
            prefix=prefix,
        )

        return test_metics

    def predict(self, data: Optional[Data] = None) -> Tensor:
        """Predict the labels for the given data, or the training data if data is None."""
        with console.status(f'moving data to {self.device}'):
            data = data.to(self.device, non_blocking=True)
        return self.link_predictor.predict(data)

    def _train(self, train_data: Data, val_data: Optional[Data] = None, test_data: Optional[Data] = None, prefix: str = '') -> Metrics:
        console.info('training link predictor')
        self.link_predictor.to(self.device)

        metrics = self.trainer.fit(
            model=self.link_predictor,
            epochs=self.epochs,
            optimizer=self.configure_optimizer(),
            train_dataloader=[train_data], 
            val_dataloader=[val_data] if val_data is not None else None,
            test_dataloader=[test_data] if test_data is not None and globals['debug'] else None,
            checkpoint=True,
            prefix=prefix,
        )

        return metrics

    def configure_trainer(self, **kwargs) -> Trainer:
        trainer = Trainer(
            monitor='val/auc', 
            monitor_mode='max', 
            **kwargs
        )
        return trainer

    def configure_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self.link_predictor.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
