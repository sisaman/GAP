import random
import time
import enum
import functools
import dgl
import numpy as np
import torch


def seed_everything(seed):
    dgl.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bootstrap(data, func=np.mean, n_boot=10000, seed=None):
    n = len(data)
    data = np.asarray(data)
    rng = np.random.default_rng(seed)
    integers = rng.integers
    
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = integers(0, n, n, dtype=np.intp)  # intp is indexing dtype
        sample = [data.take(resampler, axis=0)]
        boot_dist.append(func(*sample))
        
    return np.array(boot_dist)


def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2


def measure_runtime(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        print(f'\nTotal time spent in {str(func.__name__)}:', end - start, 'seconds.\n\n')
        return out

    return wrapper


def colored_text(msg, color):
    if isinstance(color, str):
        color = TermColors.FG.__dict__[color]
    return color.value + msg + TermColors.Control.reset.value


class Enum(enum.Enum):
    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class TermColors:
    class Control(enum.Enum):
        reset = '\033[0m'
        bold = '\033[01m'
        disable = '\033[02m'
        underline = '\033[04m'
        reverse = '\033[07m'
        strikethrough = '\033[09m'
        invisible = '\033[08m'

    class FG(enum.Enum):
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class BG(enum.Enum):
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'
