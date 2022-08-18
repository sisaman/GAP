from typing import Callable
import torch
from torch import Tensor
from torch.nn import Dropout, BatchNorm1d
from torch_geometric.nn import GraphSAGE
from torch_sparse import SparseTensor


class SAGE(GraphSAGE):
    """
    A GraphSAGE model.
    This implementation supports plain_last option.
    """
    def __init__(self, *,
                 output_dim: int,
                 hidden_dim: int = 16,  
                 num_layers: int = 2, 
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 plain_last: bool = True,
                 ):
        super().__init__(
            in_channels=-1,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=output_dim,
            dropout=dropout,
            act=activation_fn,
            norm=BatchNorm1d(hidden_dim) if batch_norm else None,
            jk=None,
            aggr='sum',
            root_weight=True,
            normalize=True,
        )
        
        self.dropout_fn = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.plain_last = plain_last
        if not plain_last and batch_norm:
            self.bn = BatchNorm1d(output_dim)
        
    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        x = super().forward(x, adj_t)
        if not self.plain_last:
            x = self.bn(x) if self.batch_norm else x
            x = self.dropout_fn(x)
            x = self.activation_fn(x)
        return x        

    def reset_parameters(self):
        super().reset_parameters()
        if not self.plain_last and self.batch_norm:
            self.bn.reset_parameters()