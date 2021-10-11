import torch
import torch.nn.functional as F
from torch.nn import AlphaDropout, SELU, ModuleList, Dropout, ReLU
from torch_geometric.nn import BatchNorm, MessagePassing, Linear
from utils import pairwise
from args import support_args


class PrivSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, perturbation_mode, privacy_mechanism, root_weight=False, cached=False):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.cached = cached
        self.agg_cached = None

        self.perturbation_mode = perturbation_mode
        self.privacy_mechanism = privacy_mechanism
        
        self.lin_l = Linear(in_channels, out_channels, bias=True)
        if self.root_weight:
            self.lin_r = Linear(in_channels, out_channels, bias=False)

    def private_aggregation(self, x, edge_index):
        if self.agg_cached is None or not self.cached:
            x = self.privacy_mechanism.clip(x, c=1)                            # todo replace with clipping

            if self.perturbation_mode == 'feature':    
                x = self.privacy_mechanism.perturb(x, sensitivity=1, account=self.training)
            
            agg = x + self.propagate(edge_index, x=x)

            if self.perturbation_mode == 'aggr':
                agg = self.privacy_mechanism.perturb(agg, sensitivity=1, account=self.training)

            self.agg_cached = agg

        return self.agg_cached

    def forward(self, x, edge_index):
        out = self.private_aggregation(x, edge_index)
        out = self.lin_l(out)

        if self.root_weight:
            out += self.lin_r(x)

        return out


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_fn, activation_fn, batchnorm, is_pred_module):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout_fn
        self.activation = activation_fn
        self.is_pred_module = is_pred_module
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim] if num_layers > 0 else []
        self.layers = ModuleList([Linear(in_channels, out_channels) for in_channels, out_channels in pairwise(dimensions)])
        self.bns = batchnorm and ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - int(is_pred_module))])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == self.num_layers - 1 and self.is_pred_module:
                break

            x = self.bns[i](x) if self.bns else x
            x = self.dropout(x)
            x = self.activation(x)
        
        return x
        

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, stage_type, 
                 dropout_fn, activation_fn, batchnorm, cache_first, is_pred_module):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.dropout = dropout_fn
        self.activation = activation_fn
        self.cache_first = cache_first
        self.is_pred_module = is_pred_module
        
        self.layers = self.init_layers()
        self.bns = batchnorm and ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - int(is_pred_module))])

    def init_layers(self) -> ModuleList:
        raise NotImplementedError

    def forward(self, x, edge_index):
        h_in = h_out = x

        for i, conv in enumerate(self.layers):
            h_out = conv(h_in, edge_index)

            if i == self.num_layers - 1 and self.is_pred_module:
                break

            h_out = self.bns[i](h_out) if self.bns else h_out
            h_out = self.dropout(h_out)
            h_out = self.activation(h_out)
            h_in = h_out = self.combine(input=h_in, output=h_out)

        return h_out

    def combine(self, input, output):
        if self.stage_type == 'stack':
            return output
        elif self.stage_type == 'skipsum':
            return input + output
        elif self.stage_type == 'skipcat':
            return torch.cat([input, output], dim=1)


class PrivateGNN(GNN):
    def __init__(self, *args, perturbation_mode, privacy_mechanism, **kwargs):
        self.perturbation_mode = perturbation_mode
        self.privacy_mechanism = privacy_mechanism
        super().__init__(*args, **kwargs)


class PrivateGraphSAGE(PrivateGNN):
    def init_layers(self):
        layers = []

        if self.num_layers > 0:

            dimensions = [self.input_dim] + (self.num_layers - 1) * [self.hidden_dim] + [self.output_dim]
            input_dim_cumulative = dimensions[0]

            for i, (in_channels, out_channels) in enumerate(pairwise(dimensions)):
                conv = PrivSAGEConv(
                    in_channels=input_dim_cumulative if self.stage_type == 'skipcat' else in_channels,
                    out_channels=out_channels,
                    perturbation_mode=self.perturbation_mode,
                    privacy_mechanism=self.privacy_mechanism,
                    cached=(i == 0 and self.cache_first),
                    root_weight=False,
                )
                layers.append(conv)
                input_dim_cumulative += out_channels
        
        return ModuleList(layers)
        

@support_args
class PrivateNodeClassifier(torch.nn.Module):
    supported_models = {
        'sage': PrivateGraphSAGE
    }

    def __init__(self,
                 num_features, num_classes, privacy_mechanism, perturbation_mode,
                 model: dict(help='base GNN model', choices=supported_models) = 'sage',
                 hidden_dim: dict(help='dimension of the hidden layers') = 16,
                 pre_layers: dict(help='number of pre-processing linear layers') = 0,
                 mp_layers: dict(help='number of message-passing layers') = 2,
                 post_layers: dict(help='number of post-processing linear layers') = 0,
                 dropout: dict(help='dropout rate (between zero and one)') = 0.0,
                 normalization: dict(help='type of NN normalization', choices=['batchnorm', 'selfnorm'], type=str) = None,
                 stage: dict(help='stage type of skip connection', choices=['stack', 'skipsum', 'skipcat']) = 'stack',
                 inductive=False,
                 ):

        super().__init__()

        if normalization == 'selfnorm':
            activaiton_fn = SELU(inplace=True)
            dropout_fn = AlphaDropout(dropout, inplace=True)
        else:
            activaiton_fn = ReLU(inplace=True)
            dropout_fn = Dropout(dropout, inplace=True)

        self.pre_mlp = MLP(
            input_dim=num_features, 
            hidden_dim=hidden_dim, 
            output_dim=num_classes if mp_layers + post_layers == 0 else hidden_dim, 
            num_layers=pre_layers, 
            activation_fn=activaiton_fn, 
            dropout_fn=dropout_fn, 
            batchnorm=normalization=='batchnorm',
            is_pred_module=(mp_layers + post_layers == 0)
        )

        GraphNN = self.supported_models[model]

        self.gnn = GraphNN(
            input_dim=num_features if pre_layers == 0 else hidden_dim, 
            hidden_dim=hidden_dim, 
            output_dim=num_classes if post_layers == 0 else hidden_dim, 
            num_layers=mp_layers, 
            stage_type=stage, 
            dropout_fn=dropout_fn, 
            activation_fn=activaiton_fn, 
            batchnorm=normalization=='batchnorm', 
            cache_first=not inductive and pre_layers == 0,
            perturbation_mode=perturbation_mode, 
            privacy_mechanism=privacy_mechanism,
            is_pred_module=(post_layers == 0),
        )

        input_dim = num_features if pre_layers + mp_layers == 0 or (pre_layers == 0 and stage == 'skipcat') else hidden_dim
        input_dim += int(stage == 'skipcat') * mp_layers * hidden_dim

        self.post_mlp = MLP(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=num_classes, 
            num_layers=post_layers, 
            activation_fn=activaiton_fn, 
            dropout_fn=dropout_fn, 
            batchnorm=normalization=='batchnorm',
            is_pred_module=True
        )

        if normalization == 'selfnorm':
            self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            # biases zero
            if len(param.shape) == 1:
                torch.nn.init.constant_(param, 0)
            # others using lecun-normal initialization
            else:
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, data):
        h = self.pre_mlp(data.x)
        h = self.gnn(h, data.edge_index)
        h = self.post_mlp(h)
        h = F.log_softmax(h, dim=1)
        return h

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
