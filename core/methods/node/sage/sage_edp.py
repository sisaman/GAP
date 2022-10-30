from typing import Annotated, Optional
import torch
from torch_geometric.data import Data
from core import console
from core.args.utils import ArgInfo
from core.methods.node import SAGE
from core.privacy.algorithms import AsymmetricRandResponse
from core.modules.base import Metrics


class EdgePrivSAGE (SAGE):
    """edge-private GraphSAGE method"""
    
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[SAGE])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.mechanism = AsymmetricRandResponse(eps=epsilon)

    def perturb_data(self, data: Data) -> Data:
        data.adj_t = data.adj_t.to(self.device)
        with console.status('perturbing graph structure'):
            data.adj_t = self.mechanism(data.adj_t, chunk_size=500)
        return data

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        data = self.perturb_data(data)
        return super().fit(data, prefix=prefix)

    def test(self, data: Optional[Data] = None, prefix: str = '') -> Metrics:
        if data is not None and data != self.data:
            data = self.perturb_data(data)
        return super().test(data, prefix=prefix)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is not None and data != self.data:
            data = self.perturb_data(data)
        return super().predict(data)
