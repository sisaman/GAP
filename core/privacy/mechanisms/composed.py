from core.privacy.mechanisms.commons import GaussianMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism
from autodp.transformer_zoo import ComposeGaussian, Composition
from autodp.mechanism_zoo import ExactGaussianMechanism


class ComposedNoisyMechanism(NoisyMechanism):
    def __init__(self, 
                 noise_scale: float, 
                 mechanism_list: list[NoisyMechanism], 
                 coeff_list: list[float]=None,
                 weight_list: list[float]=None,
                 ):
        super().__init__(noise_scale)
        if coeff_list is None:
            coeff_list = [1] * len(mechanism_list)
        if weight_list is None:
            weight_list = [1] * len(mechanism_list)
        self.params = {
            'noise_scale': noise_scale, 
            'mechanism_list': mechanism_list, 
            'coeff_list': coeff_list,
            'weight_list': weight_list
        }
        mechanism_list = [mech.update(weight * noise_scale) for mech, weight in zip(mechanism_list, weight_list)]
        mech = Composition()(mechanism_list, coeff_list)
        self.set_all_representation(mech)


class ComposedGaussianMechanism(NoisyMechanism):
    def __init__(self, 
                 noise_scale: float, 
                 mechanism_list: list[GaussianMechanism], 
                 coeff_list: list[float]
                 ):
        super().__init__(noise_scale)
        self.params = {'noise_scale': noise_scale, 'mechanism_list': mechanism_list, 'coeff_list': coeff_list}
        mechanism_list = [mech.update(noise_scale) for mech in mechanism_list]
        gm_list = [ExactGaussianMechanism(sigma=mech.params['noise_scale']) for mech in mechanism_list]
        mech = ComposeGaussian()(gm_list, coeff_list)
        self.set_all_representation(mech)