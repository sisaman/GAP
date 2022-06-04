import logging
import torch
from typing import Annotated, Literal, Optional, Union
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from pysrc.methods.base import MethodBase
from pysrc.classifiers import GraphSAGEClassifier
from pysrc.classifiers.base import Metrics, Stage


class SAGE (MethodBase):
    """non-private GraphSAGE method"""
    
    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hidden_dim:      Annotated[int,   dict(help='dimension of the hidden layers')] = 16,
                 base_layers:     Annotated[int,   dict(help='number of base MLP layers')] = 2,
                 mp_layers:       Annotated[int,   dict(help='number of GNN layers')] = 1,
                 head_layers:     Annotated[int,   dict(help='number of head MLP layers')] = 1,
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

        assert mp_layers >= 1, 'number of message-passing layers must be at least 1'
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.base_layers = base_layers
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        activation_fn = self.supported_activations[activation]

        self.classifier = GraphSAGEClassifier(
            hidden_dim=hidden_dim, 
            num_classes=num_classes, 
            base_layers=base_layers,
            mp_layers=mp_layers, 
            head_layers=head_layers, 
            normalize=False,
            activation_fn=activation_fn, 
            dropout=dropout, 
            batch_norm=batch_norm,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.classifier.reset_parameters()

    def fit(self, data: Data) -> Metrics:
        self.data = data
        metrics = self.train_classifier()
        return metrics

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is None:
            data = self.data
        return self.classifier.predict(data)

    def train_classifier(self) -> Metrics:
        logging.info('training classifier')
        self.classifier.to(self.device)

        metrics = self.trainer.fit(
            model=self.classifier, 
            epochs=self.epochs, 
            optimizer=self.configure_optimizer(),
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

    def data_loader(self, stage: Stage) -> NeighborLoader:
        if self.batch_size == 'full' or (stage != 'train' and self.full_batch_eval):
            self.data.batch_size = self.data.num_nodes
            return [self.data]
        else:
            return NeighborLoader(
                data=self.data, 
                num_neighbors=[-1]*self.mp_layers, 
                input_nodes=self.data[f'{stage}_mask'],
                replace=False,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=6,
            )

    def configure_optimizer(self) -> Optimizer:
        Optim = {'sgd': SGD, 'adam': Adam}[self.optimizer_name]
        return Optim(self.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
