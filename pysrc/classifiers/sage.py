from typing import Callable
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_sparse import SparseTensor
from pysrc.classifiers.base import ClassifierBase, Metrics, Stage
from pysrc.models import MLP
from pysrc.models import GraphSAGE
from torch_geometric.data import Data


class GraphSAGEClassifier(ClassifierBase):
    def __init__(self, 
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

        self.base_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=base_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.dropout = dropout
        self.activation_fn = activation_fn
        self.base_layers = base_layers
        self.head_layers = head_layers
        self.normalize = normalize
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = BatchNorm1d(hidden_dim)
            self.bn2 = BatchNorm1d(hidden_dim)

        self.gnn = GraphSAGE(
            in_channels=-1,
            hidden_channels=hidden_dim,
            num_layers=mp_layers,
            out_channels=num_classes if head_layers == 0 else hidden_dim,
            dropout=dropout,
            act=activation_fn,
            norm=BatchNorm1d(hidden_dim) if batch_norm else None,
            jk='last',
            aggr='add',
            root_weight=True,
            normalize=True,
        )

        self.head_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=head_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x: Tensor, adj_t: SparseTensor) -> Tensor:
        if self.base_layers > 0:
            x = self.base_mlp(x)
            x = self.bn1(x) if self.batch_norm else x
            x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
            x = self.activation_fn(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        h = self.gnn(x, adj_t)

        if self.head_layers > 0:
            h = self.bn2(h) if self.batch_norm else h
            h = F.dropout(h, p=self.dropout, training=self.training, inplace=True)
            h = self.activation_fn(h)
            h = self.head_mlp(h)

        return h

    def step(self, data: Data, stage: Stage) -> tuple[Tensor, Metrics]:
        mask = data[f'{stage}_mask']
        batch_size = data.batch_size if hasattr(data, 'batch_size') else data.num_nodes
        target = data.y[mask][:batch_size]
        h = self(data.x, data.adj_t)[mask][:batch_size]
        preds = F.log_softmax(h, dim=-1)
        acc = preds.argmax(dim=1).eq(target).float().mean() * 100
        metrics = {f'{stage}/acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=target)
            metrics[f'{stage}/loss'] = loss.detach()

        return loss, metrics

    @torch.no_grad()
    def predict(self, data: Data) -> Tensor:
        self.eval()
        h = self(data.x, data.adj_t)
        return torch.softmax(h, dim=-1)

    def reset_parameters(self):
        if self.batch_norm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

        self.base_mlp.reset_parameters()
        self.gnn.reset_parameters()
        self.head_mlp.reset_parameters()
