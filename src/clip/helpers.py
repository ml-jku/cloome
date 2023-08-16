from multiprocessing import Pool
from functools import partial


def parallelize(func, iterable, n_workers, **kwargs):
    f = partial(func, **kwargs)
    if n_workers > 1:
        with Pool(n_workers) as p:
            results = p.map(f, iterable)
    else:
        results = list(map(f, iterable))
    return results
