from typing import Callable
import torch
from torch import Tensor
from torch.nn import Dropout, ModuleList, BatchNorm1d
from torch_geometric.nn import Linear
from opacus.grad_sample import register_grad_sampler


@register_grad_sampler(Linear)
def compute_lazy_linear_grad_sample(layer, activations, backprops):
    gs = torch.einsum("n...i,n...j->nij", backprops, activations)
    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret


class MLP(torch.nn.Module):
    def __init__(self, 
                 output_dim: int,
                 hidden_dim: int = 16,  
                 num_layers: int = 2, 
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = False,
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_fn = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers = ModuleList([Linear(-1, dim) for dim in dimensions])
        
        num_bns = batch_norm * (num_layers - 1)
        self.bns = ModuleList([BatchNorm1d(hidden_dim) for _ in range(num_bns)]) if batch_norm else []
        
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.bns[i](x) if self.bns else x
                x = self.dropout_fn(x)
                x = self.activation_fn(x)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
        for bn in self.bns:
            bn.reset_parameters()