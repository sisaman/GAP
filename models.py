from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import SELU, ModuleList, Dropout, ReLU, Tanh, Module, LazyBatchNorm1d, BatchNorm1d
from opacus.grad_sample import register_grad_sampler
import torch_geometric
from torch_geometric.nn import GraphSAGE as PyGraphSAGE, Linear


supported_activations = {
    'relu': partial(ReLU, inplace=True),
    'selu': partial(SELU, inplace=True),
    'tanh': Tanh,
}


# @register_grad_sampler(LazyLinear)
# def compute_lazy_linear_grad_sample(layer, activations, backprops):
#     gs = torch.einsum("n...i,n...j->nij", backprops, activations)
#     ret = {layer.weight: gs}
#     if layer.bias is not None:
#         ret[layer.bias] = torch.einsum("n...k->nk", backprops)
#     return ret

@register_grad_sampler(torch_geometric.nn.Linear)
def compute_lazy_linear_grad_sample(layer, activations, backprops):
    gs = torch.einsum("n...i,n...j->nij", backprops, activations)
    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, activation, batch_norm):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers = ModuleList([Linear(-1, dim) for dim in dimensions])
        
        num_bns = batch_norm * (num_layers - 1)
        self.bns = ModuleList([LazyBatchNorm1d() for _ in range(num_bns)]) if batch_norm else []
        
        self.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < self.num_layers - 1:
                x = self.bns[i](x) if self.bns else x
                x = self.dropout(x)
                x = self.activation(x)

        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        
        for bn in self.bns:
            bn.reset_parameters()


class MultiStageClassifier(Module):
    supported_combinations = {
        'cat', 'sum', 'max', 'mean', 'att'   ### TODO implement attention
    }

    def __init__(self, num_stages, hidden_dim, output_dim, pre_layers, post_layers, 
                 combination_type, activation, dropout, batch_norm):

        super().__init__()
        self.combination_type = combination_type

        self.pre_mlps = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=pre_layers,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
            )] * num_stages
        )

        self.bn = LazyBatchNorm1d() if batch_norm else False
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x_stack):
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.pre_mlps)]
        h = self.combine(h_list)
        h = F.normalize(h, p=2, dim=-1)
        h = self.bn(h) if self.bn else h
        h = self.dropout(h)
        h = self.activation(h)
        h = self.post_mlp(h)
        return F.log_softmax(h, dim=-1)

    def combine(self, h_list):
        if self.combination_type == 'cat':
            return torch.cat(h_list, dim=-1)
        elif self.combination_type == 'sum':
            return torch.stack(h_list, dim=0).sum(dim=0)
        elif self.combination_type == 'mean':
            return torch.stack(h_list, dim=0).mean(dim=0)
        elif self.combination_type == 'max':
            return torch.stack(h_list, dim=0).max(dim=0).values
        elif self.combination_type == 'att':
            raise NotImplementedError
        else:
            raise ValueError(f'Unknown combination type {self.combination_type}')

    @torch.no_grad()
    def encode(self, x_stack):
        self.eval()
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.pre_mlps)]
        h_combined = self.combine(h_list)
        return h_combined

    def step(self, batch, stage):
        x_stack, y = batch
        preds = self(x_stack)
        acc = (preds.argmax(dim=1) == y).float().mean() * 100
        metrics = {f'{stage}/acc': acc}

        loss = None
        if stage != 'test':
            loss = F.nll_loss(input=preds, target=y)
            metrics[f'{stage}/loss'] = loss.detach()

        return loss, metrics

    def reset_parameters(self):
        if self.bn:
            self.bn.reset_parameters()

        for mlp in self.pre_mlps:
            mlp.reset_parameters()
        
        self.post_mlp.reset_parameters()

        if hasattr(self, 'autograd_grad_sample_hooks'):
            for hook in self.autograd_grad_sample_hooks:
                hook.remove()
            del self.autograd_grad_sample_hooks


class GraphSAGE(PyGraphSAGE):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, 
                 dropout=0.0, act=ReLU(inplace=True), norm=None, jk='last', **kwargs):
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, 
                         out_channels=out_channels, dropout=dropout, act=act, norm=norm, jk=jk, **kwargs)
        if num_layers == 1:
            self.convs = ModuleList([
                self.init_conv(in_channels, out_channels, **kwargs)
            ])


class GraphSAGEClassifier(Module):
    def __init__(self, hidden_dim, output_dim, pre_layers, mp_layers, post_layers, 
                 activation, dropout, batch_norm):

        super().__init__()

        self.pre_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=pre_layers,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.batch_norm = batch_norm
        self.bn1 = LazyBatchNorm1d() if batch_norm else False
        self.dropout1 = Dropout(dropout, inplace=True)
        self.activation1 = supported_activations[activation]()
        self.pre_layers = pre_layers

        self.gnn = GraphSAGE(
            in_channels=-1,
            hidden_channels=hidden_dim,
            num_layers=mp_layers,
            out_channels=output_dim,
            dropout=dropout,
            act=supported_activations[activation](),
            norm=BatchNorm1d(hidden_dim) if batch_norm else None,
            jk='last',
            aggr='add',
            root_weight=True,
            normalize=True,
        )

        self.bn2 = LazyBatchNorm1d() if batch_norm else False
        self.dropout2 = Dropout(dropout, inplace=True)
        self.activation2 = supported_activations[activation]()
        self.post_layers = post_layers

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x, adj_t):
        if self.pre_layers > 0:
            x = self.pre_mlp(x)
            x = self.bn1(x) if self.batch_norm else x
            x = self.dropout1(x)
            x = self.activation1(x)

        h = self.gnn(x, adj_t)

        if self.post_layers > 0:
            h = self.bn2(h) if self.batch_norm else h
            h = self.dropout2(h)
            h = self.activation2(h)
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
