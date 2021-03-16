
import numpy as np
from time import perf_counter
from contextlib import contextmanager

import tensorflow as tf


def rand(p=0.5, size=None):
    return np.random.choice([True, False], p=[p, 1 - p], size=size)


def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))


def softmin(a):
    return softmax(-a)


def discount(a, gamma):
    return gamma ** np.arange(0, len(a)) * a


def find_loss(values, actions, rewards):
    likelihood = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=values, labels=actions
    )

    loss = tf.reduce_mean(likelihood * rewards)

    return loss


def all_equal(*a):
    return all(element == a[0] for element in a)


@contextmanager
def timer(msg, transform=None):
    start = perf_counter()
    yield
    stop = perf_counter()

    duration = stop - start

    if transform is not None:
        duration = transform(duration)

    print(msg.format(duration))