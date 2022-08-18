from typing import Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from core.models import MLP
from torch_geometric.data import Data
from core.models.sage import SAGE
from core.modules.base import Metrics, Stage, TrainableModule


class SAGENodeClassifier(TrainableModule):
    def __init__(self, *,
                 num_classes: int, 
                 hidden_dim: int = 16, 
                 base_layers: int = 0, 
                 mp_layers: int = 2, 
                 head_layers: int = 0, 
                 normalize: bool = False,
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_,
                 dropout: float = 0.0, 
                 batch_norm: bool = False,
                 ):

        assert mp_layers > 0, 'Must have at least one message passing layer'
        super().__init__()
        self.normalize = normalize

        self.base_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=base_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=False,
        )

        self.gnn = SAGE(
            output_dim=num_classes if head_layers == 0 else hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=mp_layers,
            dropout=dropout,
            activation_fn=activation_fn,
            batch_norm=batch_norm,
            plain_last=head_layers == 0,
        )

        self.head_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=head_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            plain_last=True,
        )

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        x = self.base_mlp(x)
        x = F.normalize(x, p=2, dim=-1) if self.normalize else x
        x = self.gnn(x, adj_t)
        x = self.head_mlp(x)
        return x

    def step(self, data: Data, stage: Stage) -> tuple[Tensor, Metrics]:
        mask = data[f'{stage}_mask']
        h = self(data.x, data.adj_t)[mask]
        preds = F.log_softmax(h, dim=-1)
        target = data.y[mask]
        acc = preds.argmax(dim=1).eq(target).float().mean() * 100
        metrics = {'acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=target)
            metrics['loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        h = self(data.x, data.adj_t)
        return torch.softmax(h, dim=-1)

    def reset_parameters(self):
        self.base_mlp.reset_parameters()
        self.gnn.reset_parameters()
        self.head_mlp.reset_parameters()
