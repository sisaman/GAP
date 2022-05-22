import torch
import torch.nn.functional as F
from torch.nn import Module, BatchNorm1d
from pysrc.models import MLP
from pysrc.models import GraphSAGE


class GraphSAGEClassifier(Module):
    def __init__(self, hidden_dim, output_dim, pre_layers, mp_layers, post_layers, normalize,
                 activation_fn, dropout, batch_norm):

        assert mp_layers > 0, 'Must have at least one message passing layer'
        super().__init__()

        self.pre_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=pre_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.dropout = dropout
        self.activation_fn = activation_fn
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.normalize = normalize
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = BatchNorm1d(hidden_dim)
            self.bn2 = BatchNorm1d(hidden_dim)

        self.gnn = GraphSAGE(
            in_channels=-1,
            hidden_channels=hidden_dim,
            num_layers=mp_layers,
            out_channels=output_dim if post_layers == 0 else hidden_dim,
            dropout=dropout,
            act=activation_fn,
            norm=BatchNorm1d(hidden_dim) if batch_norm else None,
            jk='last',
            aggr='add',
            root_weight=True,
            normalize=True,
        )

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x, adj_t):
        if self.pre_layers > 0:
            x = self.pre_mlp(x)
            x = self.bn1(x) if self.batch_norm else x
            x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
            x = self.activation_fn(x)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        h = self.gnn(x, adj_t)

        if self.post_layers > 0:
            h = self.bn2(h) if self.batch_norm else h
            h = F.dropout(h, p=self.dropout, training=self.training, inplace=True)
            h = self.activation_fn(h)
            h = self.post_mlp(h)

        return F.log_softmax(h, dim=-1)

    def step(self, data, stage):
        mask = data[f'{stage}_mask']
        target = data.y[mask][:data.batch_size]
        adj_t = data.adj_t[:data.num_nodes, :data.num_nodes]
        preds = self(data.x, adj_t)[mask][:data.batch_size]
        acc = (preds.argmax(dim=1) == target).float().mean() * 100
        metrics = {f'{stage}/acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=target)
            metrics[f'{stage}/loss'] = loss.detach()

        return loss, metrics

    def reset_parameters(self):
        if self.batch_norm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

        self.pre_mlp.reset_parameters()
        self.gnn.reset_parameters()
        self.post_mlp.reset_parameters()

        if hasattr(self, 'autograd_grad_sample_hooks'):
            for hook in self.autograd_grad_sample_hooks:
                hook.remove()
            del self.autograd_grad_sample_hooks
