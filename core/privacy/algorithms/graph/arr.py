import torch
import numpy as np
from torch_sparse import SparseTensor


class AsymmetricRandResponse:
    def __init__(self, eps: float):
        self.eps_link = eps * 0.9
        self.eps_density = eps * 0.1

    def __call__(self, adj_t: SparseTensor, chunk_size: int=1000) -> SparseTensor:
        chunks = self.split(adj_t, chunk_size=chunk_size)
        pert_chunks = []

        for chunk in chunks:    
            chunk_pert = self.perturb(chunk)
            pert_chunks.append(chunk_pert)

        perturbed_adj_t = self.merge(pert_chunks, chunk_size=chunk_size)
        return perturbed_adj_t
    
    def split(self, adj_t: SparseTensor, chunk_size: int) -> list[SparseTensor]:
        chunks = []
        for i in range(0, adj_t.size(0), chunk_size):
            if (i + chunk_size) <= adj_t.size(0):
                chunks.append(adj_t[i:i+chunk_size])
            else:
                chunks.append(adj_t[i:])
        return chunks
    
    def perturb(self, adj_t: SparseTensor) -> SparseTensor:
        n = adj_t.size(1)
        sensitivity = 1 / (n*n)
        p = 1 / (1 + np.exp(-self.eps_link))
        d = np.random.laplace(loc=adj_t.density(), scale=sensitivity/self.eps_density)
        q = d / (2*p*d - p - d + 1)
        q = min(1, q)
        pr_1to1 = p * q
        pr_0to1 = (1 - p) * q
        mask = adj_t.to_dense(dtype=bool)
        out = mask * pr_1to1 + (~mask) * pr_0to1
        torch.bernoulli(out, out=out)
        out = SparseTensor.from_dense(out, has_value=False)
        return out
    
    def merge(self, chunks: list[SparseTensor], chunk_size: int) -> SparseTensor:
        n = (len(chunks) - 1) * chunk_size + chunks[-1].size(0)
        m = chunks[0].size(1)
        row = torch.cat([chunk.coo()[0] + i * chunk_size for i, chunk in enumerate(chunks)])
        col = torch.cat([chunk.coo()[1] for chunk in chunks])
        out = SparseTensor(row=row, col=col, sparse_sizes=(n, m))#.coalesce()
        return out
