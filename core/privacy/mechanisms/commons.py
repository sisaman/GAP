import torch
from torch import Tensor
from autodp.mechanism_zoo import Mechanism, ExactGaussianMechanism, LaplaceMechanism as AutoDPLaplaceMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism


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


class GaussianMechanism(NoisyMechanism):
    def __init__(self, noise_scale: float):
        # "noise_scale" is the std of the noise divide by the L2 sensitivity
        super().__init__(noise_scale=noise_scale)
        gm = ExactGaussianMechanism(sigma=noise_scale)
        self.name = 'GaussianMechanism'
        self.set_all_representation(gm)

    def perturb(self, data: Tensor, sensitivity: float) -> Tensor:
        std = self.params['noise_scale'] * sensitivity
        return torch.normal(mean=data, std=std) if std else data

    def __call__(self, data: Tensor, sensitivity: float) -> Tensor:
        return self.perturb(data, sensitivity)


class LaplaceMechanism(NoisyMechanism):
    def __init__(self, noise_scale: float):
        # "noise_scale" is the Laplace scale parameter divided by the L1 sensitivity
        super().__init__(noise_scale=noise_scale)
        lm = AutoDPLaplaceMechanism(b=noise_scale)
        self.name = 'LaplaceMechanism'
        self.set_all_representation(lm)

    def perturb(self, data: Tensor, sensitivity: float) -> Tensor:
        scale = self.params['noise_scale'] * sensitivity
        return torch.distributions.Laplace(loc=data, scale=scale).sample() if scale else data

    def __call__(self, data: Tensor, sensitivity: float) -> Tensor:
        return self.perturb(data, sensitivity)
