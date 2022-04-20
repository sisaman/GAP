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


class AsymmetricRandResponse:
    def __init__(self, eps):
        self.eps_link = eps * 0.9
        self.eps_density = eps * 0.1
        
    def arr(self, data: SparseTensor):
        n = data.size(1)
        sensitivity = 1 / (n*n)
        p = 1 / (1 + np.exp(-self.eps_link))
        d = np.random.laplace(loc=data.density(), scale=sensitivity/self.eps_density)
        q = d / (2*p*d - p - d + 1)
        q = min(1, q)
        pr_1to1 = p * q
        pr_0to1 = (1 - p) * q
        mask = data.to_dense(dtype=bool)
        out = mask * pr_1to1 + (~mask) * pr_0to1
        torch.bernoulli(out, out=out)
        out = SparseTensor.from_dense(out, has_value=False)
        return out

    def __call__(self, data, chunk_size=1000):
        chunks = self.split_sparse(data, chunk_size=chunk_size)
        pert_chunks = []

        for chunk in chunks:    
            chunk_pert = self.arr(chunk)
            pert_chunks.append(chunk_pert)

        data_pert = self.merge_sparse(pert_chunks, chunk_size=chunk_size)
        return data_pert
    
    @staticmethod
    def split_sparse(mat, chunk_size):
        chunks = []
        for i in range(0, mat.size(0), chunk_size):
            if (i + chunk_size) <= mat.size(0):
                chunks.append(mat[i:i+chunk_size])
            else:
                chunks.append(mat[i:])
        return chunks
    
    @staticmethod
    def merge_sparse(chunks, chunk_size):
        n = (len(chunks) - 1) * chunk_size + chunks[-1].size(0)
        m = chunks[0].size(1)
        row = torch.cat([chunk.coo()[0] + i * chunk_size for i, chunk in enumerate(chunks)])
        col = torch.cat([chunk.coo()[1] for chunk in chunks])
        out = SparseTensor(row=row, col=col, sparse_sizes=(n, m))#.coalesce()
        return out


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