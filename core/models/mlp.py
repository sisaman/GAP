from typing import Callable
import torch
from torch import Tensor
from torch.nn import Dropout, ModuleList, BatchNorm1d, Module
from torch_geometric.nn import Linear
from opacus.grad_sample import register_grad_sampler, compute_linear_grad_sample


@register_grad_sampler(Linear)
def compute_lazy_linear_grad_sample(layer, activations, backprops):
    return compute_linear_grad_sample(layer, activations, backprops)


class MLP(Module):
    """
    A multi-layer perceptron (MLP) model.
    This implementation handles 0-layer configurations as well.
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
        super().__init__()
        self.num_layers = num_layers
        self.dropout_fn = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn
        self.plain_last = plain_last

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers: list[Linear] = ModuleList([Linear(-1, dim) for dim in dimensions])
        
        num_bns = batch_norm * (num_layers - int(plain_last))
        self.bns: list[BatchNorm1d] = []
        if batch_norm:
            self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_bns)])
        
    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - self.plain_last:
                x = self.bns[i](x) if self.bns else x
                x = self.dropout_fn(x)
                x = self.activation_fn(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
        for bn in self.bns:
            bn.reset_parameters()