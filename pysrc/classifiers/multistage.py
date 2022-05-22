import torch
import torch.nn.functional as F
from torch.nn import Dropout, ModuleList, BatchNorm1d, Module
from pysrc.models import MLP


class MultiStageClassifier(Module):
    supported_combinations = {
        'cat', 'sum', 'max', 'mean', 'att'   ### TODO implement attention
    }

    def __init__(self, num_stages, hidden_dim, output_dim, pre_layers, post_layers, 
                 combination_type, normalize, activation_fn, dropout, batch_norm):

        super().__init__()
        self.combination_type = combination_type
        self.normalize = normalize

        self.pre_mlps = ModuleList([
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=pre_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
            )] * num_stages
        )

        self.bn = BatchNorm1d(hidden_dim * num_stages) if batch_norm else False
        self.dropout = Dropout(dropout, inplace=True)
        self.activation_fn = activation_fn

        self.post_mlp = MLP(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=post_layers,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
        )

    def forward(self, x_stack):
        x_stack = x_stack.permute(2, 0, 1) # (hop, batch, input_dim)
        h_list = [mlp(x) for x, mlp in zip(x_stack, self.pre_mlps)]
        h = self.combine(h_list)
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        h = self.bn(h) if self.bn else h
        h = self.dropout(h)
        h = self.activation_fn(h)
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
