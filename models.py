from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import SELU, ModuleList, Dropout, ReLU, Tanh, LazyLinear, Module, LazyBatchNorm1d
from opacus.grad_sample import register_grad_sampler


supported_activations = {
    'relu': partial(ReLU, inplace=True),
    'selu': partial(SELU, inplace=True),
    'tanh': Tanh,
}


@register_grad_sampler(LazyLinear)
def compute_lazy_linear_grad_sample(layer, activations, backprops):
    gs = torch.einsum("n...i,n...j->nij", backprops, activations)
    ret = {layer.weight: gs}
    if layer.bias is not None:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops)
    return ret


class MLP(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout, activation, batchnorm):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        dimensions = [hidden_dim] * (num_layers - 1) + [output_dim] * (num_layers > 0)
        self.layers = ModuleList([LazyLinear(dim) for dim in dimensions])
        
        num_bns = batchnorm * (num_layers - 1)
        self.bns = ModuleList([LazyBatchNorm1d() for _ in range(num_bns)]) if batchnorm else []
        
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
                 combination_type, activation, dropout, batchnorm):

        super().__init__()
        self.combination_type = combination_type

        self.pre_mlps = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=pre_layers,
                dropout=dropout,
                activation=activation,
                batchnorm=batchnorm,
            )] * num_stages
        )

        self.bn = LazyBatchNorm1d() if batchnorm else False
        self.dropout = Dropout(dropout, inplace=True)
        self.activation = supported_activations[activation]()

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation=activation,
            dropout=dropout,
            batchnorm=batchnorm,
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


