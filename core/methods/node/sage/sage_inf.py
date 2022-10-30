import torch
from typing import Annotated
from core.args.utils import ArgInfo
from core.methods.node import NodeClassification
from core.modules.node.sage import SAGENodeClassifier


class SAGE (NodeClassification):
    """Non-private GraphSAGE method"""
    
    supported_activations = {
        'relu': torch.relu_,
        'selu': torch.selu_,
        'tanh': torch.tanh,
    }

    def __init__(self,
                 num_classes,
                 hidden_dim:      Annotated[int,   ArgInfo(help='dimension of the hidden layers')] = 16,
                 base_layers:     Annotated[int,   ArgInfo(help='number of base MLP layers')] = 2,
                 mp_layers:       Annotated[int,   ArgInfo(help='number of GNN layers')] = 1,
                 head_layers:     Annotated[int,   ArgInfo(help='number of head MLP layers')] = 1,
                 activation:      Annotated[str,   ArgInfo(help='type of activation function', choices=supported_activations)] = 'selu',
                 dropout:         Annotated[float, ArgInfo(help='dropout rate')] = 0.0,
                 batch_norm:      Annotated[bool,  ArgInfo(help='if true, then model uses batch normalization')] = True,
                 **kwargs:        Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        assert mp_layers >= 1, 'number of message-passing layers must be at least 1'
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.base_layers = base_layers
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        activation_fn = self.supported_activations[activation]

        self._classifier = SAGENodeClassifier(
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

    @property
    def classifier(self) -> SAGENodeClassifier:
        return self._classifier
