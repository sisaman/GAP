import numpy as np
from scipy.stats import hypergeom
from opacus.privacy_engine import PrivacyEngine
from pysrc.privacy.mechanisms.commons import InfMechanism, ZeroMechanism
from pysrc.privacy.mechanisms.noisy import NoisyMechanism


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
