from typing import TypeVar
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from autodp.transformer_zoo import Composition, AmplificationBySampling
from torch.nn import Module
from torch.optim import Optimizer
from core.data.loader import NodeDataLoader
from core.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism

T = TypeVar('T', bound=Module)

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

    def __call__(self, module: T, optimizer: Optimizer, dataloader: NodeDataLoader) -> tuple[T, Optimizer, NodeDataLoader]:
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

    def prepare_dataloader(self, dataloader: NodeDataLoader) -> NodeDataLoader:
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            dataloader.poisson_sampling = True
        return dataloader

    def prepare_optimizer(self, optimizer: Optimizer) -> DPOptimizer:
        if self.params['noise_scale'] > 0.0 and self.params['epochs'] > 0:
            optimizer = DPOptimizer(
                optimizer=optimizer,
                noise_multiplier=self.params['noise_scale'],    # noise_multiplier is the same as noise_scale in Opacus
                max_grad_norm=self.params['max_grad_norm'],
                expected_batch_size=self.params['batch_size'],
            )
        return optimizer
