from typing import Annotated, Optional
import torch
from torch_geometric.data import Data
from pysrc.console import console
from pysrc.methods.sage.sage_inf import SAGE
from pysrc.privacy.algorithms import AsymmetricRandResponse
from pysrc.classifiers.base import Metrics


class EdgePrivSAGE (SAGE):
    """edge-private GraphSAGE method"""
    
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, dict(help='DP epsilon parameter', option='-e')],
                 **kwargs:      Annotated[dict,  dict(help='extra options passed to base class', bases=[SAGE])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.mechanism = AsymmetricRandResponse(eps=epsilon)

    def perturb_data(self, data: Data) -> Data:
        with console.status('perturbing graph structure'):
            data.adj_t = self.mechanism(data.adj_t, chunk_size=500)
        return data

    def fit(self, data: Data) -> Metrics:
        data = self.perturb_data(data)
        return super().fit(data)

    def test(self, data: Optional[Data] = None) -> Metrics:
        if data is not None and data != self.data:
            data = self.perturb_data(data)
        return super().test(data)

    def predict(self, data: Optional[Data] = None) -> torch.Tensor:
        if data is not None and data != self.data:
            data = self.perturb_data(data)
        return super().predict(data)
