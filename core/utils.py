import math
import random
from typing import Callable, Iterable, TypeVar, Optional
import numpy as np
import torch
from itertools import tee
from numpy.typing import ArrayLike, NDArray
from rich.table import Table
from rich.highlighter import ReprHighlighter
from rich import box
from tabulate import tabulate


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


def dict2table(input_dict: dict, num_cols: int = 4, title: Optional[str] = None) -> Table:
    num_items = len(input_dict)
    num_rows = math.ceil(num_items / num_cols)
    col = 0
    data = {}
    keys = []
    vals = []

    for i, (key, val) in enumerate(input_dict.items()):
        keys.append(f'{key}:')
        
        vals.append(val)
        if (i + 1) % num_rows == 0:
            data[col] = keys
            data[col+1] = vals
            keys = []
            vals = []
            col += 2

    data[col] = keys
    data[col+1] = vals

    highlighter = ReprHighlighter()
    message = tabulate(data, tablefmt='plain')
    table = Table(title=title, show_header=False, box=box.HORIZONTALS)
    table.add_row(highlighter(message))
    return table
