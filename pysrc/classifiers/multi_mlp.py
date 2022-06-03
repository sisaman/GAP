from typing import Callable, Iterable, Literal, get_args
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Dropout, ModuleList, BatchNorm1d
from pysrc.classifiers.base import ClassifierBase, Metrics, Stage
from pysrc.models import MLP


class MultiMLPClassifier(ClassifierBase):
    CombType = Literal['cat', 'sum', 'max', 'mean']
    supported_combinations = get_args(CombType)

    def __init__(self, 
                 num_inputs: int, 
                 output_dim: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 combination: CombType = 'cat',
                 normalize: bool = False, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 ):

        super().__init__()
        self.combination = combination
        self.normalize = normalize

        self.base_mlps: list[MLP] = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=base_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
            )] * num_inputs
        )

        self.bn = BatchNorm1d(hidden_dim * num_inputs) if batch_norm else False
        self.dropout = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn

        self.head_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=head_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x_stack: Tensor) -> Tensor:
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.base_mlps)]
        h = self.combine(h_list)
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        h = self.bn(h) if self.bn else h
        h = self.dropout(h)
        h = self.activation_fn(h)
        h = self.head_mlp(h)
        return F.log_softmax(h, dim=-1)

    def combine(self, h_list: Iterable[Tensor]) -> Tensor:
        if self.combination == 'cat':
            return torch.cat(h_list, dim=-1)
        elif self.combination == 'sum':
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.combination == 'mean':
            return torch.stack(h_list, dim=0).mean(dim=0)
        elif self.combination == 'max':
            return torch.stack(h_list, dim=0).max(dim=0).values
        else:
            raise ValueError(f'Unknown combination type {self.combination}')

    @torch.no_grad()
    def encode(self, x_stack: Tensor) -> Tensor:
        self.eval()
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.base_mlps)]
        h_combined = self.combine(h_list)
        return h_combined

    def step(self, batch: tuple[Tensor, Tensor], stage: Stage) -> tuple[Tensor, Metrics]:
        x_stack, y = batch
        preds: Tensor = self(x_stack)
        acc = preds.argmax(dim=1).eq(y).float().mean() * 100
        metrics = {f'{stage}/acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics[f'{stage}/loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, x_stack: Tensor) -> Tensor:
        self.eval()
        logits = self.forward(x_stack)
        return torch.exp(logits)

    def reset_parameters(self):
        if self.bn:
            self.bn.reset_parameters()

        for mlp in self.base_mlps:
            mlp.reset_parameters()
        
        self.head_mlp.reset_parameters()

