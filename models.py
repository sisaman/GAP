from privacy import GaussianMechanism
import torch
import torch.nn.functional as F
from torch.nn import AlphaDropout, SELU, ModuleList
from torch_geometric.nn import BatchNorm, MessagePassing, Linear
from utils import pairwise
from args import support_args
from loggers import Logger
import wandb


class Dense(Linear):
    def forward(self, x, _):
        return super().forward(x)


class PrivateConv(MessagePassing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturbation_mode = None
        self.mechanism: GaussianMechanism = None

    def set_privacy_mechanism(self, mechanism, perturbation_mode):
        self.mechanism = mechanism
        self.perturbation_mode = perturbation_mode


class PrivSAGEConv(PrivateConv):
    def __init__(self, in_channels, out_channels, root_weight=False, cached=False, **kwargs):
        super().__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.cached = cached
        self.cached_agg = None
        
        self.lin_l = Linear(in_channels, out_channels, bias=True)
        if self.root_weight:
            self.lin_r = Linear(in_channels, out_channels, bias=False)

    def private_aggregation(self, x, edge_index):
        if self.cached_agg is None or not self.cached:
            x = F.normalize(x, p=2., dim=-1)                            # to keep sensitivity = 1        

            if self.perturbation_mode == 'feature':    
                x = self.mechanism.perturb(x, l2_sensitivity=1)
            
            agg = x + self.propagate(edge_index, x=x)

            if self.perturbation_mode == 'aggr':
                agg = self.mechanism.perturb(agg, l2_sensitivity=1)

            self.cached_agg = agg
            
        try:
            Logger.get_instance().log_summary({'aggr': wandb.Histogram(torch.norm(self.cached_agg, p=2, dim=1).cpu())})
        except:
            pass

        return self.cached_agg

    def forward(self, x, edge_index):
        out = self.private_aggregation(x, edge_index)
        out = self.lin_l(out)

        if self.root_weight:
            out += self.lin_r(x)

        return out


class PrivateGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_mp_layers, num_post_layers, dropout, use_batchnorm):
        super().__init__()
        self.layers = self.init_layers(input_dim, hidden_dim, output_dim, num_mp_layers, num_post_layers)
        self.dropout = AlphaDropout(p=dropout)
        self.activation = SELU(inplace=True)
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bns = ModuleList([BatchNorm(hidden_dim) for _ in self.layers[:-1]])
        
        self.reset_parameters()

    def init_layers(self, input_dim, hidden_dim, output_dim, num_mp_layers, num_post_layers):
        num_layers = num_mp_layers + num_post_layers
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        layers = ModuleList(self.init_message_passing_layers(dimensions[:num_mp_layers+1]))
        layers.extend([Dense(in_channels, out_channels) for in_channels, out_channels in pairwise(dimensions[num_mp_layers:])])

        return layers

    def init_message_passing_layers(self, dimensions) -> ModuleList:
        raise NotImplementedError

    def reset_parameters(self):
        for param in self.layers.parameters():
            # biases zero
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, edge_index)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x

    def set_privacy_mechanism(self, mechanism, perturbation_mode):
        for layer in self.layers:
            if isinstance(layer, PrivateConv):
                layer.set_privacy_mechanism(mechanism, perturbation_mode)


class PrivateGraphSAGE(PrivateGNN):
    def init_message_passing_layers(self, dimensions):
        return [
            PrivSAGEConv(
                in_channels=dimensions[i], 
                out_channels=dimensions[i+1],
                root_weight=False,
                cached=(i==0)
            ) 
            for i in range(len(dimensions) - 1)
        ]
    

@support_args
class PrivateNodeClassifier(torch.nn.Module):
    SupportedModels = {
        'sage': PrivateGraphSAGE
    }

    def __init__(self,
                 input_dim,
                 num_classes,
                 model: dict(help='base GNN model', choices=SupportedModels.keys()) = 'sage',
                 hidden_dim: dict(help='dimension of the hidden layers') = 32,
                 mp_layers: dict(help='number of message-passing layers') = 1,
                 fc_layers: dict(help='number of fully-connected layers') = 1,
                 batchnorm: dict(help='enables batch-normalization') = False,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 ):

        super().__init__()
        GNN = self.SupportedModels[model]
        self.gnn = GNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_mp_layers=mp_layers,
            num_post_layers=fc_layers,
            dropout=dropout,
            use_batchnorm=batchnorm,
        )

    def set_privacy_mechanism(self, mechanism, perturbation_mode):
        self.gnn.set_privacy_mechanism(mechanism, perturbation_mode)

    def forward(self, data):
        h = self.gnn(data.x, data.edge_index)
        y = F.log_softmax(h, dim=1)
        return y

    def training_step(self, data):
        y_true = data.y[data.train_mask]
        y_pred = self(data)[data.train_mask]
        
        loss = F.nll_loss(input=y_pred, target=y_true)
        acc = self.accuracy(input=y_pred, target=y_true)
        
        metrics = {'train/loss': loss.item(), 'train/acc': acc}
        return loss, metrics

    def validation_step(self, data):
        y_pred = self(data)
        y_true = data.y

        val_mask = data.val_mask
        test_mask = data.test_mask

        metrics = {
            'val/loss': F.nll_loss(input=y_pred[val_mask], target=y_true[val_mask]).item(),
            'val/acc': self.accuracy(input=y_pred[val_mask], target=y_true[val_mask]),
            'test/acc': self.accuracy(input=y_pred[test_mask], target=y_true[test_mask]),
        }

        return metrics

    @staticmethod
    def accuracy(input, target):
        input = input.argmax(dim=1) if len(input.size()) > 1 else input
        target = target.argmax(dim=1) if len(target.size()) > 1 else target
        return (input == target).float().mean().item() * 100
