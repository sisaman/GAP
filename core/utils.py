import random
from typing import Callable, Iterable, TypeVar, Optional
import numpy as np
import torch
from itertools import tee
from numpy.typing import ArrayLike, NDArray

RT = TypeVar('RT')

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise(iterable: Iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def bootstrap(data: ArrayLike, 
              func: Callable[[ArrayLike], NDArray]=np.mean, 
              n_boot: int=10000, 
              seed: Optional[int]=None
              ) -> NDArray:
    n = len(data)
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    integers = rng.integers
    
    boot_dist = []
    for _ in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [data.take(resampler, axis=0)]
        boot_dist.append(func(*sample))
        
    return np.array(boot_dist)


def confidence_interval(data: ArrayLike, 
                        func: Callable[[ArrayLike], NDArray]=np.mean, 
                        size: int=1000, 
                        ci: int=95, 
                        seed: Optional[int]=None
                        ) -> float:
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2

