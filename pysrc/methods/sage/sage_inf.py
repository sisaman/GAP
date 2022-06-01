import torch
from typing import Annotated, Literal, Union
from torch.optim import Adam, SGD, Optimizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from pysrc.methods.base import MethodBase
from pysrc.trainer import Trainer
from pysrc.classifiers import GraphSAGEClassifier
from pysrc.classifiers.base import Metrics, Stage


class SAGEINF (MethodBase):
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
                 device:          Annotated[str,   dict(help='device to use', choices=['cpu', 'cuda'])] = 'cuda',
                 epochs:          Annotated[int,   dict(help='number of epochs for training')] = 100,
                 batch_size:      Annotated[Union[Literal['full'], int],   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 use_amp:         Annotated[bool,  dict(help='use automatic mixed precision training')] = False,
                 ):

        assert mp_layers >= 1, 'number of message-passing layers must be at least 1'

        self.base_layers = base_layers
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        self.use_amp = use_amp
        activation_fn = self.supported_activations[activation]

        self.classifier = GraphSAGEClassifier(
            hidden_dim=hidden_dim, 
            output_dim=num_classes, 
            base_layers=base_layers,
            mp_layers=mp_layers, 
            head_layers=head_layers, 
            normalize=False,
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
        self.classifier.reset_parameters()
        self.trainer.reset()

    def fit(self, data: Data) -> Metrics:
        self.data = data
        metrics = self.train_classifier()
        return metrics

    def train_classifier(self) -> Metrics:
        self.classifier.to(self.device)

        metrics = self.trainer.fit(
            model=self.classifier, 
            epochs=self.epochs, 
            optimizer=self.configure_optimizer(),
            train_dataloader=self.data_loader('train'), 
            val_dataloader=self.data_loader('val'),
            test_dataloader=self.data_loader('test'),
            checkpoint=False,
        )

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
