from opacus.privacy_engine import PrivacyEngine
import torch
import numpy as np
import torch.nn.functional as F
import autodp.mechanism_zoo as mechanisms
from autodp.transformer_zoo import ComposeGaussian, Composition, AmplificationBySampling
from torch_geometric.utils import remove_self_loops, add_self_loops, contains_self_loops, coalesce
from scipy.optimize import minimize_scalar
from torch_sparse import SparseTensor
from scipy.stats import hypergeom

from data import PoissonDataLoader


class ZeroMechanism(mechanisms.Mechanism):
    def __init__(self):
        super().__init__()
        self.name = 'ZeroMechanism'
        self.params = {}
        self.propagate_updates(func=lambda _: 0, type_of_update='RDP')
        

class InfMechanism(mechanisms.Mechanism):
    def __init__(self):
        super().__init__()
        self.name = 'InfMechanism'
        self.params = {}


class NoisyMechanism(mechanisms.Mechanism):
    def __init__(self, noise_scale):
        # "noise_scale" is the std of the noise divide by the sensitivity
        super().__init__()
        self.name = 'NoisyMechanism'
        self.params = {'noise_scale': noise_scale}

    def is_zero(self):
        eps = np.array([self.RenyiDP(alpha) for alpha in range(2,100)])
        return np.all(eps == 0.0)

    def is_inf(self):
        eps = np.array([self.RenyiDP(alpha) for alpha in range(2,100)])
        return np.all(eps == np.inf)

    def update(self, noise_scale):
        self.params.pop('noise_scale')
        self.__init__(noise_scale, **self.params)
        return self

    def calibrate(self, eps, delta):
        if self.params['noise_scale'] == 0:
            self.update(noise_scale=1)  # to avoid is_inf being true

        if eps == np.inf or self.is_inf() or self.is_zero():
            self.update(noise_scale=0)
            return 0.0
        else:
            fn_err = lambda x: abs(eps - self.update(x).get_approxDP(delta))

            # check if the mechanism is already calibrated
            if fn_err(self.params['noise_scale']) < 1e-3:
                return self.params['noise_scale']

            results = minimize_scalar(fn_err, method='bounded', bounds=[0,1000], tol=1e-8, options={'maxiter': 1000000})

            if results.success and results.fun < 1e-3:
                self.update(results.x)
                return results.x
            else:
                raise RuntimeError(f"eps_delta_calibrator fails to find a parameter:\n{results}")


class GaussianMechanism(NoisyMechanism):
    def __init__(self, noise_scale):
        # "noise_scale" is the std of the noise divide by the L2 sensitivity
        super().__init__(noise_scale=noise_scale)
        gm = mechanisms.ExactGaussianMechanism(sigma=noise_scale)
        self.name = 'GaussianMechanism'
        self.set_all_representation(gm)

    def perturb(self, data, sensitivity):
        std = self.params['noise_scale'] * sensitivity
        return torch.normal(mean=data, std=std) if std else data

    def __call__(self, data, sensitivity):
        return self.perturb(data, sensitivity)


class LaplaceMechanism(NoisyMechanism):
    def __init__(self, noise_scale):
        # "noise_scale" is the Laplace scale parameter divided by the L1 sensitivity
        super().__init__(noise_scale=noise_scale)
        lm = mechanisms.LaplaceMechanism(b=noise_scale)
        self.name = 'LaplaceMechanism'
        self.set_all_representation(lm)

    def perturb(self, data, sensitivity):
        scale = self.params['noise_scale'] * sensitivity
        return torch.distributions.Laplace(loc=data, scale=scale).sample() if scale else data

    def __call__(self, data, sensitivity):
        return self.perturb(data, sensitivity)


class ComposedNoisyMechanism(NoisyMechanism):
    def __init__(self, noise_scale, mechanism_list, coeff_list):
        super().__init__(noise_scale)
        self.params = {'noise_scale': noise_scale, 'mechanism_list': mechanism_list, 'coeff_list': coeff_list}
        mechanism_list = [mech.update(noise_scale) for mech in mechanism_list]
        mech = Composition()(mechanism_list, coeff_list)
        self.set_all_representation(mech)


class ComposedGaussianMechanism(NoisyMechanism):
    def __init__(self, noise_scale, mechanism_list, coeff_list):
        super().__init__(noise_scale)
        self.params = {'noise_scale': noise_scale, 'mechanism_list': mechanism_list, 'coeff_list': coeff_list}
        mechanism_list = [mech.update(noise_scale) for mech in mechanism_list]
        gm_list = [mechanisms.ExactGaussianMechanism(sigma=mech.params['noise_scale']) for mech in mechanism_list]
        mech = ComposeGaussian()(gm_list, coeff_list)
        self.set_all_representation(mech)


class TopMFilter(NoisyMechanism):
    def __init__(self, noise_scale):
        super().__init__(noise_scale)
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

    def __call__(self, data):
        if self.params['noise_scale'] == 0.0:
            return data

        is_sparse = hasattr(data, 'adj_t')
        if is_sparse:
            adj_t = data.adj_t
            edge_index = torch.cat(adj_t.t().coo()[:-1]).view(2,-1)
        else:
            edge_index = data.edge_index

        # if the graph has self loops, we need to remove them to exclude them from edge count
        has_self_loops = contains_self_loops(edge_index)
        if has_self_loops:
            edge_index, _ = remove_self_loops(edge_index)

        n = data.num_nodes
        m = edge_index.shape[1]
        m_pert = self.mech_count.perturb(m, sensitivity=1)
        m_pert = round(m_pert.item())
        eps_edges = 1 / self.mech_edges.params['noise_scale']
        
        theta = np.log((n * (n-1) / m_pert) - 1) / (2 * eps_edges) + 0.5
        if theta > 1:
            theta = np.log((n * (n-1) / (2 * m_pert)) + 0.5 * (np.exp(eps_edges) - 1)) / eps_edges

        loc = torch.ones_like(edge_index[0]).float()
        sample = self.mech_edges.perturb(loc, sensitivity=1)
        edges_to_be_kept = edge_index[:, sample > theta]

        # we add self loops to the graph to exclude them from perturbation
        # so we will have extra n edges
        edges_with_self_loops, _ = add_self_loops(edges_to_be_kept, num_nodes=n)

        # we aim to have m_pert + n edges, so we keep adding edges until we have that many.
        while True:
            num_remaining_edges = m_pert + n - edges_with_self_loops.size(1)

            if num_remaining_edges <= 0:
                break

            edges_to_be_added = torch.randint(n, size=(2, num_remaining_edges), device=edge_index.device)
            edges_with_self_loops = torch.cat([edges_with_self_loops, edges_to_be_added], dim=1)
            edges_with_self_loops = coalesce(edges_with_self_loops, num_nodes=n, reduce='max')

        edge_index = edges_with_self_loops
        # if the input graph did not initially have self loops, we need to remove them
        if not has_self_loops: 
            edge_index, _ = remove_self_loops(edge_index)

        if is_sparse:
            data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(n, n)).t()
        else:
            data.edge_index = edge_index

        return data


class NoisySGD(NoisyMechanism):
    def __init__(self, noise_scale, dataset_size, batch_size, epochs, max_grad_norm):
        super().__init__(noise_scale)
        self.name = 'NoisySGD'
        self.params = {
            'noise_scale': noise_scale, 
            'dataset_size': dataset_size, 
            'batch_size': batch_size, 
            'epochs': epochs,
            'max_grad_norm': max_grad_norm,
        }

        if epochs == 0:
            mech = ZeroMechanism()
            self.params['noise_scale'] = 0.0
        elif noise_scale == 0.0:
            mech = InfMechanism()
        else:
            subsample = AmplificationBySampling()
            compose = Composition()
            gm = GaussianMechanism(noise_scale=noise_scale)
            subsampled_gm = subsample(gm, prob=batch_size/dataset_size, improved_bound_flag=True)
            mech = compose([subsampled_gm],[epochs * dataset_size // batch_size])
        
        self.set_all_representation(mech)

    def __call__(self, module, optimizer, data_loader, **kwargs):
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            _, optimizer, data_loader = PrivacyEngine().make_private(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.params['noise_scale'],
                max_grad_norm=self.params['max_grad_norm'],
                poisson_sampling=not isinstance(data_loader, PoissonDataLoader),
                **kwargs
            )

        return module, optimizer, data_loader


class GNNBasedNoisySGD(NoisyMechanism):
    def __init__(self, noise_scale, dataset_size, batch_size, epochs, max_grad_norm, max_degree):
        super().__init__(noise_scale)
        self.name = 'NoisySGD'
        self.params = {
            'noise_scale': noise_scale, 
            'dataset_size': dataset_size, 
            'batch_size': batch_size, 
            'epochs': epochs,
            'max_grad_norm': max_grad_norm,
            'max_degree': max_degree,
        }

        if epochs == 0:
            mech = ZeroMechanism()
            self.params['noise_scale'] = 0.0
            self.set_all_representation(mech)
        elif noise_scale == 0.0:
            mech = InfMechanism()
            self.set_all_representation(mech)
        else:
            N = dataset_size
            K = max_degree
            m = batch_size
            C = max_grad_norm
            T = epochs * dataset_size // batch_size
            DeltaK = 2 * (K + 1) * C
            
            def RDP(alpha):
                sigma = noise_scale * DeltaK
                expected_fn = lambda rho: np.exp(alpha * (alpha-1) * 2 * rho**2 * C**2 / sigma**2)
                expected_rho = hypergeom(N, K+1, m).expect(expected_fn)
                gamma = np.log(expected_rho) / (alpha - 1)
                return gamma * T

            self.propagate_updates(RDP, type_of_update='RDP')

    def __call__(self, module, optimizer, data_loader, **kwargs):
        noise_scale = self.params['noise_scale']
        epochs = self.params['epochs']
        K = self.params['max_degree']
        C = self.params['max_grad_norm']

        if noise_scale > 0.0 and epochs > 0:
            _, optimizer, data_loader = PrivacyEngine().make_private(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=noise_scale * 2 * (K + 1),
                max_grad_norm=C,
                poisson_sampling=False,
                **kwargs
            )

        return module, optimizer, data_loader


class PMA(NoisyMechanism):
    def __init__(self, noise_scale, hops):
        super().__init__(noise_scale)
        self.name = 'PMA'
        self.params = {'noise_scale': noise_scale, 'hops': hops}
        self.gm = GaussianMechanism(noise_scale=noise_scale)

        if hops == 0:
            mech = ZeroMechanism()
            self.params['noise_scale'] = 0.0
        elif noise_scale == 0.0:
            mech = InfMechanism()
        else:
            mech = ComposedGaussianMechanism(
                noise_scale=noise_scale, 
                mechanism_list=[self.gm], 
                coeff_list=[hops]
            )

        self.set_all_representation(mech)

    def __call__(self, data, sensitivity):
        assert hasattr(data, 'adj_t')

        x = F.normalize(data.x, p=2, dim=-1)
        x_list = [x]

        for _ in range(1, self.params['hops'] + 1):
            # aggregate
            x = data.adj_t.matmul(x)
            # perturb
            x = self.gm(x, sensitivity=sensitivity)
            # normalize
            x = x = F.normalize(x, p=2, dim=-1)

            x_list.append(x)

        data.x = torch.stack(x_list, dim=-1)
        return data
