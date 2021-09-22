import torch
import numpy as np
from autodp.mechanism_zoo import ExactGaussianMechanism
from torch_geometric.utils import remove_self_loops, add_self_loops


class GaussianMechanism:
    def __init__(self, noise_std, delta):
        self.noise_std = noise_std
        self.delta = delta
        self.sigma_list = []

    def perturb(self, data, l2_sensitivity):
        if self.noise_std > 0:
            sigma = self.noise_std / l2_sensitivity
            self.sigma_list.append(sigma)
            data = torch.normal(mean=data, std=self.noise_std)

        return data

    def get_privacy_spent(self):
        if not self.sigma_list:
            return 1e9-1
        sigma_list = np.array(self.sigma_list)
        sigma_total = np.sqrt(1/np.sum(1/sigma_list**2))
        mechanism = ExactGaussianMechanism(sigma=sigma_total)
        epsilon = mechanism.get_approxDP(self.delta)
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

    @staticmethod
    def to_sparse_adjacency(edge_index, num_nodes):
        return torch.sparse_coo_tensor(
            indices=edge_index,
            values=torch.ones_like(edge_index[0]).float(),
            size=(num_nodes, num_nodes),
            device=edge_index.device
        )
