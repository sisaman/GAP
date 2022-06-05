from typing_extensions import Self
from scipy.optimize import minimize_scalar, OptimizeResult
from autodp.mechanism_zoo import Mechanism
import numpy as np


class NoisyMechanism(Mechanism):
    def __init__(self, noise_scale: float):
        # "noise_scale" is the std of the noise divide by the sensitivity
        super().__init__()
        self.name = 'NoisyMechanism'
        self.params = {'noise_scale': noise_scale}

    def update(self, noise_scale: float) -> Self:
        self.params.pop('noise_scale')
        self.__init__(noise_scale, **self.params)
        return self

    def calibrate(self, eps: float, delta: float) -> float:
        if self.params['noise_scale'] == 0:
            self.update(noise_scale=1)  # to avoid is_inf being true

        if np.isinf(eps):
            self.update(noise_scale=0)
            return 0.0
        else:
            fn_err = lambda x: abs(eps - self.update(np.exp(x)).get_approxDP(delta))

            # check if the mechanism is already calibrated
            if fn_err(self.params['noise_scale']) < 1e-3:
                return self.params['noise_scale']
            
            result: OptimizeResult = minimize_scalar(fn_err, 
                method='brent', 
                options={'xtol': 1e-5, 'maxiter': 1000000}
            )

            if result.success and result.fun < 1e-3:
                noise_scale = np.exp(result.x)
                self.update(noise_scale)
                return noise_scale
            else:
                raise RuntimeError(f"calibrator failed to find noise scale\n{result}")