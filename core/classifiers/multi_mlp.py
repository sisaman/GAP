from typing import Callable, Iterable, Literal, get_args
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, LazyBatchNorm1d, Linear
from core.classifiers.mlp import MLPClassifier
from core.models import MLP


class MultiMLPClassifier(MLPClassifier):
    CombType = Literal['cat', 'sum', 'max', 'mean', 'att']
    supported_combinations = get_args(CombType)

    def __init__(self, 
                 num_inputs: int, 
                 num_classes: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 combination: CombType = 'cat',
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
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

        self.base_mlps: list[MLP] = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=base_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
            ) for _ in range(num_inputs)]
        )

        self.bn = LazyBatchNorm1d() if batch_norm else False
        self.combination = combination
        if combination == 'att':
            self.hidden_dim = hidden_dim
            self.num_heads = num_inputs
            self.Q = Linear(in_features=hidden_dim, out_features=self.num_heads, bias=False)

    def forward(self, x_stack: Tensor) -> Tensor:
        x_stack = x_stack.permute(2, 0, 1) # (hop, node, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.base_mlps)]
        h = self.combine(h_list)
        h = F.normalize(h, p=2, dim=-1)
        h = self.bn(h) if self.bn else h
        h = self.dropout_fn(h)
        h = self.activation_fn(h)
        h = super().forward(h)
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
        if self.bn: self.bn.reset_parameters()
        for mlp in self.base_mlps: mlp.reset_parameters()
        if self.combination == 'att':
            self.Q.reset_parameters()
        super().reset_parameters()
