from typing_extensions import Self
from scipy.optimize import minimize_scalar, OptimizeResult
from autodp.mechanism_zoo import Mechanism
import numpy as np
from core import console


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

    def is_zero(self) -> bool:
        for alpha in range(2,100):
            if self.RenyiDP(alpha) > 0:
                return False
        return True

    def is_inf(self) -> bool:
        for alpha in range(2,100):
            if not np.isinf(self.RenyiDP(alpha)):
                return False
        return True

    def calibrate(self, eps: float, delta: float) -> float:
        if self.params['noise_scale'] == 0:
            self.update(noise_scale=1)  # to avoid is_inf being true
        
        console.debug('checking if the mechanism is inf or zero...')
        if np.isinf(eps) or self.is_inf() or self.is_zero():
            self.update(noise_scale=0)
            return 0.0
        else:
            console.debug('calibration begins...')
            fn_err = lambda x: abs(eps - self.update(np.exp(x)).get_approxDP(delta))

            # check if the mechanism is already calibrated
            if fn_err(self.params['noise_scale']) < 1e-3:
                return self.params['noise_scale']
            
            result: OptimizeResult = minimize_scalar(fn_err, 
                method='brent', 
                options={'xtol': 1e-5, 'maxiter': 1000000, 'disp': 0}
            )

            if result.success and result.fun < 1e-3:
                noise_scale = np.exp(result.x)
                self.update(noise_scale)
                return noise_scale
            else:
                raise RuntimeError(f"calibrator failed to find noise scale\n{result}")