from functools import partial
from autodp.transformer_zoo import AmplificationBySampling, ComposeGaussian, Composition
import torch
import torch.nn.functional as F
from torch.nn import SELU, ModuleList, Dropout, ReLU, Tanh
from torch_geometric.nn import BatchNorm, MessagePassing, Linear, MessageNorm
from torch_geometric.utils import add_remaining_self_loops
from utils import pairwise
from args import support_args
from privacy import Calibrator, GaussianMechanism, NullMechanism, TopMFilter, supported_mechanisms


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_fn, activation_fn, batchnorm, output_module):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout_fn
        self.activation = activation_fn
        self.output_module = output_module
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim] if num_layers > 0 else []
        self.layers = ModuleList([Linear(in_channels, out_channels) for in_channels, out_channels in pairwise(dimensions)])
        self.bns = ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - int(output_module))]) if batchnorm else []
        self.reset_parameters()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == self.num_layers - 1 and self.output_module:
                break

            x = self.bns[i](x) if self.bns else x
            x = self.dropout(x)
            x = self.activation(x)
        
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        

class GNN(torch.nn.Module):
    supported_stages = {'stack', 'skipsum', 'skipmax', 'skipcat'}

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, stage_type, 
                 dropout_fn, activation_fn, batchnorm, root_weight, inductive, input_module, output_module):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.dropout = dropout_fn
        self.activation = activation_fn
        self.root_weight = root_weight
        self.inductive = inductive
        self.input_module = input_module
        self.output_module = output_module
        
        self.layers = self.init_layers()
        self.bns = ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - int(output_module))]) if batchnorm else []
        self.reset_parameters()

    def init_layers(self) -> ModuleList:
        raise NotImplementedError

    def layer_forward(self, index, h_in, edge_index):
        h_out = self.layers[index](h_in, edge_index)

        if index == self.num_layers - 1 and self.output_module:
            return h_out

        h_out = self.bns[index](h_out) if self.bns else h_out
        h_out = self.dropout(h_out)
        h_out = self.activation(h_out)
        
        return h_out

    def forward(self, x, edge_index):
        h_in = h_out = x
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        for i in range(self.num_layers):
            h_out = self.layer_forward(i, h_in, edge_index)
            h_in = h_out = self.combine(input=h_in, output=h_out)

        return h_out

    def combine(self, input, output):
        if self.stage_type == 'skipsum' and input.size() == output.size():
            return input + output
        elif self.stage_type == 'skipmax' and input.size() == output.size():
            return torch.max(input, output)
        elif self.stage_type == 'skipcat':
            return torch.cat([input, output], dim=1)
        else:
            return output

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


class PrivConv(MessagePassing):
    def __init__(self, in_channels, out_channels, root_weight, cached, perturbation, mechanism):
        super().__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.cached = cached
        self.cached_agg = None
        self.perturbation_mode = perturbation
        self.mechanism = mechanism
        
        self.lin_l = Linear(in_channels, out_channels, bias=True)
        if self.root_weight:
            self.lin_r = Linear(in_channels, out_channels, bias=False)        

        self.mn = MessageNorm(learn_scale=True)
        self.reset_parameters()

    def private_aggregation(self, x, edge_index):
        if self.cached_agg is None or not self.cached:
            
            if self.perturbation_mode in {'aggr', 'feature'}:
                x = self.mechanism.normalize(x)

            if self.perturbation_mode == 'feature':    
                x = self.mechanism.perturb(x, sensitivity=1)
            
            agg = self.propagate(edge_index, x=x)

            if self.perturbation_mode == 'aggr':
                agg = self.mechanism.perturb(agg, sensitivity=1)

            self.cached_agg = agg

        return self.cached_agg

    def forward(self, x, edge_index):
        out = self.private_aggregation(x, edge_index)
        out = self.mn(x, out)
        out = self.lin_l(out)

        if self.root_weight:
            out += self.lin_r(x)

        out = F.normalize(out, p=2, dim=-1)
        return out

    def reset_parameters(self):
        self.cached_agg = None
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        self.mn.reset_parameters()


class PrivateGNN(GNN):
    supported_perturbations = {'aggr', 'feature', 'graph'}                               ###### shoud be changed ######

    def __init__(self, perturbation, mechanism, *args, **kwargs):
        self.perturbation = perturbation
        
        if perturbation == 'graph':
            self.base_mechanism = TopMFilter(noise_scale=0.0)
        else:
            self.base_mechanism = supported_mechanisms[mechanism](noise_scale=0.0)

        self.cached_edge_index = None

        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        self.cached_edge_index = None
        return super().reset_parameters()

    def forward(self, x, edge_index):
        if self.cached_edge_index is None or self.inductive:
            if self.perturbation == 'graph':
                edge_index = self.base_mechanism.perturb(edge_index, num_nodes=x.size(0))
            self.cached_edge_index = edge_index
            
        return super().forward(x, self.cached_edge_index)

    def update(self, noise_scale):
        self.base_mechanism.update(noise_scale)

    def build_mechanism(self, epochs, sampling_rate, noise_scale=None):
        if noise_scale is None:
            noise_scale = self.base_mechanism.noise_scale

        if self.num_layers == 0 or noise_scale == 0.0:
            return NullMechanism()
            
        self.update(noise_scale=noise_scale)
        compose = ComposeGaussian() if isinstance(self.base_mechanism, GaussianMechanism) else Composition()

        if sampling_rate == 1.0:
            if self.perturbation == 'graph':
                return self.base_mechanism
            else:
                num_calls = int(self.cachable_first_agg) + (self.num_layers - int(self.cachable_first_agg)) * epochs
                composed_mech = compose([self.base_mechanism], [num_calls])
                return composed_mech
        else:
            if self.perturbation == 'graph':
                subsample = AmplificationBySampling(PoissonSampling=True)
                subsampled_mech = subsample(self, prob=sampling_rate, improved_bound_flag=True)
                complex_mech = Composition()([subsampled_mech], [epochs])
                return complex_mech
            else:
                model_mech = compose([self.base_mechanism], [self.num_layers])
                subsample = AmplificationBySampling(PoissonSampling=True)
                subsampled_mech = subsample(model_mech, prob=sampling_rate, improved_bound_flag=True)
                complex_mech = Composition()([subsampled_mech], [epochs])
                return complex_mech

    def init_layers(self):
        layers = []
        mechanism = self.base_mechanism if self.perturbation in {'aggr', 'feature'} else None

        for i in range(self.num_layers):
            conv = PrivConv(
                in_channels=-1,
                out_channels=self.output_dim if i == self.num_layers - 1 else self.hidden_dim,
                cached=(i == 0 and self.cachable_first_agg),
                root_weight=self.root_weight,
                perturbation=self.perturbation,
                mechanism=mechanism
            )
            layers.append(conv)
        
        return ModuleList(layers)

    @property
    def cachable_first_agg(self):
        return (not self.inductive) and self.input_module
        

@support_args
class PrivateNodeClassifier(torch.nn.Module):
    supported_activations = {
        'relu': partial(ReLU, inplace=True), 
        'selu': partial(SELU, inplace=True), 
        'tanh': Tanh,
    }

    def __init__(self,
                 num_classes, 
                 perturbation:  dict(help='perturbation method', option='-p', choices=PrivateGNN.supported_perturbations) = 'aggr', 
                 mechanism:     dict(help='perturbation mechanism', choices=supported_mechanisms) = 'gaussian', 
                 hidden_dim:    dict(help='dimension of the hidden layers') = 16,
                 pre_layers:    dict(help='number of pre-processing linear layers') = 0,
                 mp_layers:     dict(help='number of message-passing layers') = 2,
                 post_layers:   dict(help='number of post-processing linear layers') = 0,
                 activation:    dict(help='type of activation function', choices=supported_activations) = 'relu',
                 dropout:       dict(help='dropout rate (between zero and one)') = 0.0,
                 batchnorm:     dict(help='if True, then model uses batch normalization') = True,
                 stage:         dict(help='stage type of skip connection', choices=GNN.supported_stages) = 'stack',
                 root_weight:   dict(help='if True, the layer adds transformed root node features to the output.') = True,
                 inductive=False,
                 ):

        super().__init__()

        self.privacy_accountant = None
        self.activaiton_fn = self.supported_activations[activation]()
        dropout_fn = Dropout(dropout, inplace=True)

        self.pre_mlp = MLP(
            input_dim=-1, 
            hidden_dim=hidden_dim, 
            output_dim=num_classes if mp_layers + post_layers == 0 else hidden_dim, 
            num_layers=pre_layers, 
            activation_fn=self.activaiton_fn, 
            dropout_fn=dropout_fn, 
            batchnorm=batchnorm,
            output_module=(mp_layers + post_layers == 0)
        )

        self.gnn = PrivateGNN(
            input_dim=-1, 
            hidden_dim=hidden_dim, 
            output_dim=num_classes if post_layers == 0 else hidden_dim, 
            num_layers=mp_layers, 
            stage_type=stage, 
            dropout_fn=dropout_fn, 
            activation_fn=self.activaiton_fn, 
            batchnorm=batchnorm, 
            root_weight=root_weight,
            inductive=inductive,
            input_module=(pre_layers == 0),
            output_module=(post_layers == 0),
            perturbation=perturbation,
            mechanism=mechanism,
        )

        self.post_mlp = MLP(
            input_dim=-1, 
            hidden_dim=hidden_dim, 
            output_dim=num_classes, 
            num_layers=post_layers, 
            activation_fn=self.activaiton_fn, 
            dropout_fn=dropout_fn, 
            batchnorm=batchnorm,
            output_module=True
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.activaiton_fn.__init__()
        self.pre_mlp.reset_parameters()
        self.gnn.reset_parameters()
        self.post_mlp.reset_parameters()

    def forward(self, data):
        h = self.pre_mlp(data.x)
        h = self.gnn(h, data.edge_index)
        h = self.post_mlp(h)
        h = F.log_softmax(h, dim=1)
        return h

    @torch.no_grad()
    def embed(self, data):
        self.eval()
        x = self.pre_mlp(data.x)
        data.x = x
        return data

    def training_step(self, data, epoch):
        y_true = data.y[data.train_mask]
        y_pred = self(data)[data.train_mask]
        
        loss = F.nll_loss(input=y_pred, target=y_true)
        acc = self.accuracy(input=y_pred, target=y_true)
        metrics = {'train/loss': loss.item(), 'train/acc': acc}

        if self.privacy_accountant:
            metrics['eps'] = self.privacy_accountant(epochs=epoch)

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

    def calibrate(self, epsilon, delta, epochs, sampling_rate):
        mechanism_builder = lambda noise_scale: self.gnn.build_mechanism(
            noise_scale=noise_scale, 
            epochs=epochs, 
            sampling_rate=sampling_rate
        )

        noise_scale = Calibrator(mechanism_builder).calibrate(eps=epsilon, delta=delta)
        self.gnn.update(noise_scale=noise_scale)
        return self

    def init_privacy_accountant(self, delta, sampling_rate):
        self.privacy_accountant = lambda epochs: self.gnn.build_mechanism(
            epochs=epochs,
            sampling_rate=sampling_rate
        ).get_approxDP(delta=delta)
