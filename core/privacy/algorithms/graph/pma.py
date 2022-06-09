import torch
from core.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from core.privacy.mechanisms.composed import ComposedGaussianMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism


class PMA(NoisyMechanism):
    def __init__(self, noise_scale: float, hops: int):
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

    def __call__(self, x: torch.Tensor, sensitivity: float) -> torch.Tensor:
        return self.gm(x, sensitivity=sensitivity)
