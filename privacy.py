import torch
import numpy as np
import torch.nn.functional as F
import autodp.mechanism_zoo as mechanisms
from autodp.transformer_zoo import Composition, AmplificationBySampling
from torch_geometric.utils import remove_self_loops, add_self_loops
from args import support_args


class GaussianMechanism(mechanisms.ExactGaussianMechanism):
    def __init__(self, sigma):
        super().__init__(sigma=sigma)

    def perturb(self, data, sensitivity):
        std = self.params['sigma'] * sensitivity
        return torch.normal(mean=data, std=std)


class LaplaceMechanism(mechanisms.LaplaceMechanism):
    def __init__(self, sigma):
        super().__init__(b=sigma)

    def perturb(self, data, sensitivity):
        scale = self.params['b'] * sensitivity
        return torch.distributions.Laplace(loc=data, scale=scale).sample()
    

@support_args
class NoisyMechanism:

    supported_mechanisms = {
        'laplace': LaplaceMechanism,
        'gaussian': GaussianMechanism
    }

    def __init__(self, 
                 mechanism:     dict(help='perturbation mechanism', option='-m', choices=supported_mechanisms) = 'gaussian',
                 noise_scale :  dict(help='scale parameter of the noise', option='-n', type=float) = None, 
                 delta:         dict(help='DP delta parameter', option='-d') = 1e-6,
                 sampling_prob = 1.0
    ):
        self.sigma = noise_scale
        self.delta = delta
        self.sampling_prob = sampling_prob
        self.perturb_count = 0

        subsample = AmplificationBySampling(PoissonSampling=True)
        self.mechanism = self.supported_mechanisms[mechanism](self.sigma)
        self.subsampled_mechanism = subsample(self.mechanism, self.sampling_prob, improved_bound_flag=True)
        self.compose = Composition()

    def perturb(self, data, sensitivity, account=True):
        if self.sigma is not None:
            data = self.mechanism.perturb(data, sensitivity=sensitivity)

            if account: 
                self.perturb_count += 1

        return data

    def normalize(self, data):
        return F.normalize(data, p=(1 if self.mechanism is LaplaceMechanism else 2), dim=-1)

    def get_privacy_spent(self):
        if self.perturb_count == 0:
            return 1e10-1

        composed_mechanism = self.compose(
            mechanism_list=[self.subsampled_mechanism],
            coeff_list=[self.perturb_count]
        )

        epsilon = composed_mechanism.get_approxDP(self.delta)
        return epsilon


class TopMFilter:
    def __init__(self, eps_edges, eps_count):
        self.eps_edges = eps_edges
        self.eps_count = eps_count

    def perturb(self, data):
        if self.eps_edges == np.inf:
            return data

        data.edge_index, _ = remove_self_loops(data.edge_index)
        n = data.num_nodes
        m = data.num_edges
        m_pert = torch.distributions.Laplace(loc=m, scale=1/self.eps_count).sample()
        m_pert = round(m_pert.item())

        theta = np.log((n * (n-1) / m_pert) - 1) / (2 * self.eps_edges) + 0.5
        if theta > 1:
            theta = np.log((n * (n-1) / (2 * m_pert)) + 0.5 *
                           (np.exp(self.eps_edges) - 1)) / self.eps_edges

        loc = torch.ones_like(data.edge_index[0]).float()
        sample = torch.distributions.Laplace(
            loc, scale=1/self.eps_edges).sample()
        edges_to_be_removed = data.edge_index[:, sample < theta]

        edge_index_with_self_loops, _ = add_self_loops(
            data.edge_index, num_nodes=data.num_nodes)
        adjmat = self.to_sparse_adjacency(
            edge_index_with_self_loops, num_nodes=n)
        ### adjmat has m+n entries ###

        while True:
            adjmat = adjmat.coalesce()
            nnz = adjmat.values().size(0)
            num_remaining_edges = m_pert + n + \
                edges_to_be_removed.size(1) - nnz

            if num_remaining_edges <= 0:
                break

            edges_to_be_added = torch.randint(
                n, size=(2, num_remaining_edges), device=adjmat.device)
            adjmat = adjmat + \
                self.to_sparse_adjacency(edges_to_be_added, num_nodes=n)

        adjmat = (adjmat.bool().int() -
                  self.to_sparse_adjacency(edges_to_be_removed, num_nodes=n)).coalesce()
        edge_index, values = adjmat.indices(), adjmat.values()
        data.edge_index = edge_index[:, values > 0].contiguous()
        data.edge_index, _ = remove_self_loops(data.edge_index)
        return data

    def get_privacy_spent(self):
        return self.eps_count + self.eps_edges

    @staticmethod
    def to_sparse_adjacency(edge_index, num_nodes):
        return torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones_like(edge_index[0]).float(),
            size=(num_nodes, num_nodes),
            device=edge_index.device
        )
