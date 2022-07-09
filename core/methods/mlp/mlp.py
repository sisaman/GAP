import torch
from typing import Annotated, Literal, Union
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from core.methods.base import NodeClassificationBase
from core.classifiers import MLPClassifier
from core.classifiers.base import Stage


class MLP (NodeClassificationBase):
    """Non-private MLP method"""

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
                 batch_size:      Annotated[Union[Literal['full'], int],   
                                                   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 **kwargs:        Annotated[dict,  dict(help='extra options passed to base class', bases=[NodeClassificationBase])]
                 ):

        assert num_layers > 1, 'number of layers must be greater than 1'
        super().__init__(num_classes, **kwargs)

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        activation_fn = self.supported_activations[activation]

        self._classifier = MLPClassifier(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def classifier(self) -> MLPClassifier:
        return self._classifier

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
