from scipy.optimize import minimize_scalar
from autodp.mechanism_zoo import Mechanism
import numpy as np


class NoisyMechanism(Mechanism):
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