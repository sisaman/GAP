from torch.nn import ReLU, ModuleList
from torch_geometric.nn import GraphSAGE as PyGraphSAGE


class GraphSAGE(PyGraphSAGE):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels=None, 
                 dropout=0.0, act=ReLU(inplace=True), norm=None, jk='last', **kwargs):
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, 
                         out_channels=out_channels, dropout=dropout, act=act, norm=norm, jk=jk, **kwargs)
        if num_layers == 1:
            self.convs = ModuleList([
                self.init_conv(in_channels, out_channels, **kwargs)
            ])
