from typing import Callable, Iterable, Literal, get_args
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, LazyBatchNorm1d, Linear, Module, Dropout
from core.models import MLP


class MultiMLP(Module):
    CombType = Literal['cat', 'sum', 'max', 'mean', 'att']
    supported_combinations = get_args(CombType)

    def __init__(self, *, 
                 num_channels: int, 
                 output_dim: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 combination: CombType = 'cat',
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 plain_last: bool = True,
                 ):

        super().__init__()

        self.combination = combination
        self.activation_fn = activation_fn
        self.dropout_fn = Dropout(dropout, inplace=True)

        self.base_mlps: list[MLP] = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=base_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                plain_last=True,
            ) for _ in range(num_channels)]
        )

        if combination == 'att':
            self.hidden_dim = hidden_dim
            self.num_heads = num_channels
            self.Q = Linear(in_features=hidden_dim, out_features=self.num_heads, bias=False)

        self.bn = LazyBatchNorm1d() if batch_norm else False

        self.head_mlp = MLP(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=head_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=plain_last,
        )

    def forward(self, x_stack: Tensor) -> Tensor:
        x_stack = x_stack.permute(2, 0, 1) # (hop, node, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.base_mlps)]
        h = self.combine(h_list)
        h = F.normalize(h, p=2, dim=-1)
        h = self.bn(h) if self.bn else h
        h = self.dropout_fn(h)
        h = self.activation_fn(h)
        h = self.head_mlp(h)
        return h

    def combine(self, h_list: Iterable[Tensor]) -> Tensor:
        if self.combination == 'cat':
            return torch.cat(h_list, dim=-1)
        elif self.combination == 'sum':
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.combination == 'mean':
            return torch.stack(h_list, dim=0).mean(dim=0)
        elif self.combination == 'max':
            return torch.stack(h_list, dim=0).max(dim=0).values
        elif self.combination == 'att':
            H = torch.stack(h_list, dim=1)  # (node, hop, dim)
            W = F.leaky_relu(self.Q(H), 0.2).softmax(dim=0)  # (node, hop, head)
            out = H.transpose(1, 2).matmul(W).view(-1, self.hidden_dim * self.num_heads)
            return out
        else:
            raise ValueError(f'Unknown combination type {self.combination}')

    def reset_parameters(self):
        for mlp in self.base_mlps: mlp.reset_parameters()
        if self.combination == 'att':
            self.Q.reset_parameters()
        if self.bn: self.bn.reset_parameters()
        self.head_mlp.reset_parameters()
