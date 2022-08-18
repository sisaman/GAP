from typing import Callable
import torch
from torch import Tensor
from core.models.multi_mlp import MultiMLP
from core.modules.node.mlp import MLPNodeClassifier


class ClassificationModule(MLPNodeClassifier):
    def __init__(self, *,
                 num_channels: int, 
                 num_classes: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 combination: MultiMLP.CombType = 'cat',
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 ):

        super().__init__(num_classes=num_classes)  # this is dummy, but necessary

        self.model = MultiMLP(
            num_channels=num_channels,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            base_layers=base_layers,
            head_layers=head_layers,
            combination=combination,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )
