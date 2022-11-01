from typing import Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from core.models import MLP
from torch_geometric.data import Data
from core.modules.node.mlp import MLPNodeClassifier


class EncoderModule(MLPNodeClassifier):
    def __init__(self, *,
                 num_classes: int,
                 hidden_dim: int = 16,  
                 encoder_layers: int = 2, 
                 head_layers: int = 1, 
                 normalize: bool = True,
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):

        super().__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
        )

        self.encoder_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=encoder_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )

        self.dropout_fn = Dropout(p=dropout, inplace=True)
        self.activation_fn = activation_fn
        self.normalize = normalize
        self.bn = BatchNorm1d(hidden_dim) if batch_norm else False

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        x = self.bn(x) if self.bn else x
        x = self.dropout_fn(x)
        x = self.activation_fn(x)
        x = super().forward(x)
        return x

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        x = data.x
        x = self.encoder_mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x
        
    def reset_parameters(self):
        super().reset_parameters()
        self.encoder_mlp.reset_parameters()
        if self.bn:
            self.bn.reset_parameters()