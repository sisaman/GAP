from typing import Callable, Optional, Union
from torch.nn import ModuleList, Module
from torch_geometric.nn import GraphSAGE as PyGraphSAGE


class GraphSAGE(PyGraphSAGE):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: Optional[int] = None,
                 dropout: float = 0.0,
                 act: Union[str, Callable, None] = "relu",
                 norm: Optional[Module] = None,
                 jk: Optional[str] = None,
                 **kwargs
                 ):
        super().__init__(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, 
                         out_channels=out_channels, dropout=dropout, act=act, norm=norm, jk=jk, **kwargs)
        if num_layers == 1:
            self.convs = ModuleList([
                self.init_conv(in_channels, out_channels, **kwargs)
            ])
