import numpy as np
from typing import Annotated
from torch_geometric.data import Data
from pysrc.console import console
from pysrc.methods.sage.sage_inf import SAGEINF
from pysrc.privacy.algorithms import AsymmetricRandResponse
from pysrc.classifiers.base import Metrics


class SAGEEDP (SAGEINF):
    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, dict(help='DP epsilon parameter', option='-e')] = np.inf,
                 **kwargs:      Annotated[dict, dict(help='extra options passed to GAPINF method', bases=[SAGEINF])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.mechanism = AsymmetricRandResponse(eps=epsilon)
    
    def fit(self, data: Data) -> Metrics:
        with console.status('perturbing graph structure'):
            data.adj_t = self.mechanism(data.adj_t, chunk_size=500)
        return super().fit(data)