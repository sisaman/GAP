from core.methods.base import MethodBase, NodeClassificationBase
from core.methods.gap import GAP
from core.methods.gap import EdgePrivGAP
from core.methods.gap import NodePrivGAP
from core.methods.sage import SAGE
from core.methods.sage import EdgePrivSAGE
from core.methods.sage import NodePrivSAGE
from core.methods.mlp import MLP
from core.methods.mlp import PrivMLP


supported_methods = {
    'gap-inf':  GAP,
    'gap-edp':  EdgePrivGAP,
    'gap-ndp':  NodePrivGAP,
    'sage-inf': SAGE,
    'sage-edp': EdgePrivSAGE,
    'sage-ndp': NodePrivSAGE,
    'mlp':      MLP,
    'mlp-dp':   PrivMLP
}
