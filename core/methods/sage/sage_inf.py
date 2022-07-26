import torch
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from core.methods.base import NodeClassification
from core.classifiers import GraphSAGEClassifier
from core.classifiers.base import Stage


class SAGE (NodeClassification):
    """Non-private GraphSAGE method"""
    
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
                 batch_size:      Annotated[Union[Literal['full'], int],   
                                                   dict(help='batch size, or "full" for full-batch training')] = 'full',
                 full_batch_eval: Annotated[bool,  dict(help='if true, then model uses full-batch evaluation')] = True,
                 **kwargs:        Annotated[dict,  dict(help='extra options passed to base class', bases=[NodeClassification])]
                 ):

        assert mp_layers >= 1, 'number of message-passing layers must be at least 1'
        super().__init__(num_classes=num_classes, **kwargs)
        
        self.base_layers = base_layers
        self.mp_layers = mp_layers
        self.head_layers = head_layers
        self.batch_size = batch_size
        self.full_batch_eval = full_batch_eval
        activation_fn = self.supported_activations[activation]

        self._classifier = GraphSAGEClassifier(
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
    def classifier(self) -> GraphSAGEClassifier:
        return self._classifier

    def data_loader(self, data: Data, stage: Stage) -> NeighborLoader:
        if self.batch_size == 'full' or (stage != 'train' and self.full_batch_eval):
            return [data]
        else:
            return NeighborLoader(
                data=data, 
                num_neighbors=[-1]*self.mp_layers, 
                input_nodes=data[f'{stage}_mask'],
                replace=False,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=6,
            )
