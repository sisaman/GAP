from opacus.privacy_engine import PrivacyEngine
import torch
import numpy as np
import torch.nn.functional as F
import autodp.mechanism_zoo as mechanisms
from autodp.transformer_zoo import ComposeGaussian, Composition, AmplificationBySampling
from torch_geometric.utils import remove_self_loops, add_self_loops, contains_self_loops
from scipy.optimize import minimize_scalar
from torch_sparse import SparseTensor


class Mechanism(mechanisms.Mechanism):
    def __init__(self):
        super().__init__()

    def is_zero(self):
        eps = np.array([self.RenyiDP(alpha) for alpha in range(2,100)])
        return np.all(eps == 0.0)

    def is_inf(self):
        eps = np.array([self.RenyiDP(alpha) for alpha in range(2,100)])
        return np.all(eps == np.inf)

    def calibrate(self, eps, delta):
        self.update(noise_scale=1)

        if eps == np.inf or self.is_inf() or self.is_zero():
            return 0.0
        else:
            fn_err = lambda x: abs(eps - self.update(x).get_approxDP(delta))
            results = minimize_scalar(fn_err, method='bounded', bounds=[0,1000], tol=1e-8, options={'maxiter': 1000000})

            if results.success and results.fun < 1e-3:
                self.update(results.x)
                return results.x
            else:
                raise RuntimeError(f"eps_delta_calibrator fails to find a parameter:\n{results}")


class ZeroMechanism(Mechanism):
    def __init__(self):
        super().__init__()
        self.name = 'ZeroMechanism'
        self.params = {}
        self.propagate_updates(func=lambda _: 0, type_of_update='RDP')
        

class InfMechanism(Mechanism):
    def __init__(self):
        super().__init__()
        self.name = 'InfMechanism'
        self.params = {}


class GaussianMechanism(mechanisms.ExactGaussianMechanism):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        super().__init__(sigma=noise_scale)

    def perturb(self, data, sensitivity):
        std = self.params['sigma'] * sensitivity
        return torch.normal(mean=data, std=std) if std else data

    def __call__(self, data, sensitivity):
        return self.perturb(data, sensitivity)


class LaplaceMechanism(mechanisms.LaplaceMechanism):
    def __init__(self, noise_scale):
        self.noise_scale = noise_scale
        super().__init__(b=noise_scale)

    def perturb(self, data, sensitivity):
        scale = self.params['b'] * sensitivity
        return torch.distributions.Laplace(loc=data, scale=scale).sample() if scale else data

    def __call__(self, data, sensitivity):
        return self.perturb(data, sensitivity)


class TopMFilter(Mechanism):
    def __init__(self, noise_scale):
        super().__init__()
        self.name = 'TopMFilter'
        self.params = {'noise_scale': noise_scale}

        if noise_scale == 0:
            mech = InfMechanism()
        else:
            # noise scales are set such that 0.1 of privacy budget is 
            #   spent on perturbing total edge count and 0.9 on perturbing edges
            self.mech_edges = LaplaceMechanism(noise_scale)
            self.mech_count = LaplaceMechanism(noise_scale * 9)
            mech = Composition()([self.mech_edges, self.mech_count], [1,1])
        
        self.set_all_representation(mech)

    def update(self, noise_scale):
        self.__init__(noise_scale)
        return self
 
    def __call__(self, edge_index_or_adj_t, num_nodes):
        if self.params['noise_scale'] == 0.0:
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


class NoisySGD(Mechanism):
    def __init__(self, noise_scale, dataset_size, batch_size, epochs):
        super().__init__()
        self.name = 'NoisySGD'
        self.params = {
            'noise_scale': noise_scale, 
            'dataset_size': dataset_size, 
            'batch_size': batch_size, 
            'epochs': epochs,
        }

        if epochs == 0:
            mech = ZeroMechanism()
        elif noise_scale == 0.0:
            mech = InfMechanism()
        else:
            subsample = AmplificationBySampling()
            compose = Composition()
            gm = GaussianMechanism(noise_scale=noise_scale)
            subsampled_gm = subsample(gm, prob=batch_size/dataset_size, improved_bound_flag=True)
            mech = compose([subsampled_gm],[epochs * dataset_size / batch_size])
        
        self.set_all_representation(mech)

    def update(self, noise_scale):
        self.__init__(
            noise_scale, 
            self.params['dataset_size'], 
            self.params['batch_size'], 
            self.params['epochs'], 
        )
        return self

    def __call__(self, module, optimizer, dataloader, max_grad_norm):
        if self.params['noise_scale'] == 0.0 or self.params['epochs'] == 0:
            return module, optimizer, dataloader

        return PrivacyEngine().make_private(
            module=module,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.params['noise_scale'],
            max_grad_norm=max_grad_norm,
        )


class PMA(Mechanism):
    def __init__(self, noise_scale, hops):
        super().__init__()
        self.name = 'PMA'
        self.params = {'noise_scale': noise_scale, 'hops': hops}
        self.gm = GaussianMechanism(noise_scale=noise_scale)

        if hops == 0:
            mech = ZeroMechanism()
        elif noise_scale == 0.0:
            mech = InfMechanism()
        else:
            mech = ComposeGaussian()([self.gm], [hops])

        self.set_all_representation(mech)

    def update(self, noise_scale):
        self.__init__(noise_scale, self.params['hops'])
        return self

    def __call__(self, data, sensitivity):
        assert hasattr(data, 'adj_t')

        if hasattr(data, 'x_list'):
            x = data.x_list[0]
        else:
            x = data.x

        x = F.normalize(x, p=2, dim=-1)
        data.x_list = [x]

        for _ in range(1, self.params['hops'] + 1):
            # aggregate
            x = data.adj_t.matmul(x)
            # perturb
            x = self.gm(x, sensitivity=sensitivity)
            # normalize
            x = x = F.normalize(x, p=2, dim=-1)

            data.x_list.append(x)

        return data
