from pysrc.privacy.mechanisms.noisy import NoisyMechanism
from autodp.transformer_zoo import ComposeGaussian, Composition
from autodp.mechanism_zoo import ExactGaussianMechanism


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
        gm_list = [ExactGaussianMechanism(sigma=mech.params['noise_scale']) for mech in mechanism_list]
        mech = ComposeGaussian()(gm_list, coeff_list)
        self.set_all_representation(mech)