import torch
import torch.nn.functional as F
from pysrc.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from pysrc.privacy.mechanisms.composed import ComposedGaussianMechanism
from pysrc.privacy.mechanisms.noisy import NoisyMechanism


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
