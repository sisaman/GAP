from opacus.privacy_engine import PrivacyEngine
from autodp.transformer_zoo import Composition, AmplificationBySampling
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from pysrc.data.loader.poisson import PoissonDataLoader
from pysrc.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from pysrc.privacy.mechanisms.noisy import NoisyMechanism


class NoisySGD(NoisyMechanism):
    def __init__(self, noise_scale: float, dataset_size: int, batch_size: int, epochs: int, max_grad_norm: float):
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

    def __call__(self, module: Module, optimizer: Optimizer, data_loader: DataLoader, **kwargs):
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
