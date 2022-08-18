import torch
from typing import Annotated
from core.methods.base import NodeClassification
from core.classifiers import MLPClassifier


class MLP (NodeClassification):
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
                 **kwargs:        Annotated[dict,  dict(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        assert num_layers > 1, 'number of layers must be greater than 1'
        super().__init__(num_classes, **kwargs)

        self.num_layers = num_layers
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
