from typing import TypeVar
import numpy as np
from scipy.stats import hypergeom
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.privacy_engine import forbid_accumulation_hook
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from core.privacy.mechanisms.commons import InfMechanism, ZeroMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism


T = TypeVar('T', bound=Module)


class GNNBasedNoisySGD(NoisyMechanism):
    def __init__(self, noise_scale: float, dataset_size: int, batch_size: int, 
                 epochs: int, max_grad_norm: float, max_degree: int):
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

    def __call__(self, module: T, optimizer: Optimizer, dataloader: DataLoader) -> tuple[T, Optimizer, DataLoader]:
        module = self.prepare_module(module)
        dataloader = self.prepare_dataloader(dataloader)
        optimizer = self.prepare_optimizer(optimizer)
        return module, optimizer, dataloader

    def prepare_module(self, module: T) -> T:
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            if hasattr(module, 'autograd_grad_sample_hooks'):
                for hook in module.autograd_grad_sample_hooks:
                    hook.remove()
                del module.autograd_grad_sample_hooks
            GradSampleModule(module).register_backward_hook(forbid_accumulation_hook)
        return module

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        # since we don't need poisson sampling, we can use the same dataloader
        return dataloader

    def prepare_optimizer(self, optimizer: Optimizer) -> DPOptimizer:
        noise_scale = self.params['noise_scale']
        epochs = self.params['epochs']
        K = self.params['max_degree']
        C = self.params['max_grad_norm']

        if noise_scale > 0.0 and epochs > 0:
            optimizer = DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=noise_scale * 2 * (K + 1),
                max_grad_norm=C,
                expected_batch_size=self.params['batch_size'],
            )
        return optimizer
