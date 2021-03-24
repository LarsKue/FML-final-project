
import numpy as np
from time import perf_counter
from contextlib import contextmanager


def random(p=0.5, size=None):
    return np.random.choice([True, False], p=[p, 1 - p], size=size)


def discount(a, gamma):
    return gamma ** np.arange(len(a)) * a


def returns(a, gamma):
    return np.cumsum(discount(a, gamma)[::-1])[::-1]


def normalize(a):
    result = a + np.amin(a)
    return result / np.sum(result)


def all_equal(*a):
    return all(element == a[0] for element in a)


def rolling_mean(x, n):
    return np.convolve(x, np.ones(n), "valid") / n


@contextmanager
def timer(msg, transform=None):
    start = perf_counter()
    yield
    stop = perf_counter()

    duration = stop - start

    if transform is not None:
        duration = transform(duration)

    print(msg.format(duration))


def function_timer(msg=None, transform=None):
    if transform is None:
        def _t(d):
            return 1000 * d

        transform = _t

    def outer(f):
        nonlocal msg
        if msg is None:
            msg = f"Execution of {f.__name__} took {{:.2f}} ms"

        def inner(*args, **kwargs):
            with timer(msg, transform):
                rv = f(*args, **kwargs)
            return rv
        return inner
    return outer
