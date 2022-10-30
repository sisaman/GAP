import torch
from typing import Annotated
from core.args.utils import ArgInfo
from core.methods.link.base import LinkPrediction
from core.modules.link.mlp import MLPLinkPredictor


class MLPLinkPredictionMethod (LinkPrediction):
    """Non-private MLP method for link prediction"""

    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 hidden_dim:      Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 num_layers:      Annotated[int,   ArgInfo(help='number of MLP layers')] = 2,
                 activation:      Annotated[str,   ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[LinkPrediction])]
                 ):

        assert num_layers > 1, 'number of layers must be greater than 1'
        super().__init__(**kwargs)

        self.num_layers = num_layers
        activation_fn = self.supported_activations[activation]

        self._predictor = MLPLinkPredictor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    @property
    def link_predictor(self) -> MLPLinkPredictor:
        return self._predictor
