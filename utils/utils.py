import torch
import numpy as np
from time import time


def timer(func):
    """Check time.

    Usage)
    >>> @timer
    >>> def method1(...):
    >>>     ...
    >>>     return
    """

    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Elapsed time[{func.__name__}]: {end - start} sec", flush=True)
        return result
        return func(*args, **kwargs)

    return wrapper


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    return
