import logging
import torch
import numpy as np
import torch.nn.functional as F
from autodp.autodp_core import Mechanism
import autodp.mechanism_zoo as mechanisms
from autodp.transformer_zoo import Composition
from torch_geometric.utils import remove_self_loops, add_self_loops, contains_self_loops
from scipy.optimize import minimize_scalar
from torch_sparse import SparseTensor


class NullMechanism(Mechanism):
    def __init__(self):
        super().__init__()

    def update(self, _):
        self.__init__()


class GaussianMechanism(mechanisms.ExactGaussianMechanism):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        super().__init__(sigma=noise_scale)

    def update(self, noise_scale):
        self.__init__(noise_scale)

    def perturb(self, data, sensitivity):
        std = self.params['sigma'] * sensitivity
        return torch.normal(mean=data, std=std) if std else data

    def normalize(self, data):
        if self.noise_scale == 0:
            return data
        else:
            return F.normalize(data, p=2, dim=-1)

    def clip(self, data, c):
        if self.noise_scale == 0:
            return data
        else:
            return (c / data.norm(p=2, dim=-1, keepdim=True)).clamp(max=1) * data

    def get_approxDP(self, delta):
        if self.noise_scale == 0:
            return np.inf
        else:
            return super().get_approxDP(delta)


class LaplaceMechanism(mechanisms.LaplaceMechanism):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        super().__init__(b=noise_scale)

    def update(self, noise_scale):
        self.__init__(noise_scale)

    def perturb(self, data, sensitivity):
        scale = self.params['b'] * sensitivity
        return torch.distributions.Laplace(loc=data, scale=scale).sample() if scale else data

    def normalize(self, data):
        if self.noise_scale == 0:
            return data
        else:
            return F.normalize(data, p=1, dim=-1)

    def clip(self, data, c):
        if self.noise_scale == 0:
            return data
        else:
            return (c / data.norm(p=1, dim=-1, keepdim=True)).clamp(max=1) * data

    def get_approxDP(self, delta):
        if self.noise_scale == 0:
            return np.inf
        else:
            return super().get_approxDP(delta)
    

supported_mechanisms = {
    'laplace': LaplaceMechanism,
    'gaussian': GaussianMechanism
}


class TopMFilter(Mechanism):
    def __init__(self, noise_scale):
        super().__init__()
        self.noise_scale = noise_scale
        # noise scales are set such that 0.1 of privacy budget is 
        #   spent on perturbing total edge count and 0.9 on perturbing edges
        self.mech_edges = LaplaceMechanism(self.noise_scale)
        self.mech_count = LaplaceMechanism(self.noise_scale * 9)
        composed_mech = Composition()([self.mech_edges, self.mech_count], [1,1])
        self.set_all_representation(composed_mech)

    def update(self, noise_scale):
        self.__init__(noise_scale)

    def get_approxDP(self, delta):
        if self.noise_scale == 0:
            return np.inf
        else:
            return super().get_approxDP(delta)
 
    def perturb(self, edge_index_or_adj_t, num_nodes):
        if self.noise_scale == 0.0:
            return edge_index_or_adj_t

        is_sparse = isinstance(edge_index_or_adj_t, SparseTensor)
        if is_sparse:
            adj_t = edge_index_or_adj_t
            edge_index = torch.cat(adj_t.coo()[:-1]).view(2,-1)
        else:
            edge_index = edge_index_or_adj_t

        # if the graph has self loops, we need to remove them to exclude them from edge count
        has_self_loops = contains_self_loops(edge_index)
        if has_self_loops:
            edge_index, _ = remove_self_loops(edge_index)

        n = num_nodes
        m = edge_index.shape[1]
        m_pert = self.mech_count.perturb(m, sensitivity=1)
        m_pert = round(m_pert.item())
        eps_edges = 1 / self.mech_edges.noise_scale
        
        theta = np.log((n * (n-1) / m_pert) - 1) / (2 * eps_edges) + 0.5
        if theta > 1:
            theta = np.log((n * (n-1) / (2 * m_pert)) + 0.5 * (np.exp(eps_edges) - 1)) / eps_edges

        loc = torch.ones_like(edge_index[0]).float()
        sample = self.mech_edges.perturb(loc, sensitivity=1)
        edges_to_be_removed = edge_index[:, sample < theta]

        # we add self loops to the graph to exclude them from perturbation
        edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=n)
        # adjmat will have m+n entries after adding self loops
        adjmat = self.to_sparse_adjacency(edge_index_with_self_loops, num_nodes=n)

        while True:
            adjmat = adjmat.coalesce()
            nnz = adjmat.values().size(0)

            # we want the adjacency matrix to have (m_pert + n + num_edges_to_be_removed) non-zero entries,
            #   so we keep adding edges until we have that many.
            #   we later remove (n + num_edges_to_be_removed) edges so the final graph has exactly m_pert edges
            num_remaining_edges = m_pert + n + edges_to_be_removed.size(1) - nnz

            if num_remaining_edges <= 0:
                break

            edges_to_be_added = torch.randint(n, size=(2, num_remaining_edges), device=adjmat.device)
            adjmat = adjmat + self.to_sparse_adjacency(edges_to_be_added, num_nodes=n)

        # remove edges to be removed: the graph will have m_pert + n edges
        adjmat = (adjmat.bool().int() - self.to_sparse_adjacency(edges_to_be_removed, num_nodes=n)).coalesce()
        edge_index, values = adjmat.indices(), adjmat.values()
        edge_index = edge_index[:, values > 0].contiguous()
        # if the input graph did not initially have self loops, we need to remove them
        if not has_self_loops: 
            edge_index, _ = remove_self_loops(edge_index)

        if is_sparse:
            return SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
        else:
            return edge_index

    @staticmethod
    def to_sparse_adjacency(edge_index, num_nodes):
        return torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones_like(edge_index[0]).float(),
            size=(num_nodes, num_nodes),
            device=edge_index.device
        )


class Calibrator:
    def __init__(self, mechanism_cls):
        self.mechanism_cls = mechanism_cls

    def calibrate(self, eps, delta):
        mechanism = self.mechanism_cls(noise_scale=1)

        if eps == np.inf or isinstance(mechanism, NullMechanism):
            return 0.0
        else:
            noise_scale = self.eps_delta_calibrator(eps, delta)
            logging.info(f'noise scale: {noise_scale:.4f}\n')
            return noise_scale
    
    def eps_delta_calibrator(self, eps, delta):
        fn_err = lambda x: abs(eps - self.mechanism_cls(x).get_approxDP(delta))
        results = minimize_scalar(fn_err, method='bounded', bounds=[0,1000], tol=1e-8, options={'maxiter': 1000000})

        if results.success and results.fun < 1e-3:
            return results.x
        else:
            raise RuntimeError(f"eps_delta_calibrator fails to find a parameter:\n{results}")
