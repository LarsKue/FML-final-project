
import numpy as np
from time import perf_counter
from contextlib import contextmanager


def rand(p=0.5, size=None):
    return np.random.choice([True, False], p=[p, 1 - p], size=size)


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))


def softmin(a):
    return softmax(-a)


@contextmanager
def timer(msg, transform=None):
    start = perf_counter()
    yield
    stop = perf_counter()

    duration = stop - start

    if transform is not None:
        duration = transform(duration)

    print(msg.format(duration))