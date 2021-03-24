
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
    return gamma ** np.arange(len(a)) * a


def returns(a, gamma):
    return np.cumsum(discount(a, gamma)[::-1])[::-1]


# def find_loss(logits, actions, returns):
#     """
#     :param logits: Model logits (predictions)
#     :param actions: Current model response
#     :param returns: Discounted Sum of Rewards in the remainder of the episode
#     """
#     # target response is computed using current guess for Q
#     # optimize Q using cross entropy loss, with returns as weights
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#         logits=logits, labels=actions
#     )
#
#     loss = tf.reduce_mean(loss * returns)
#
#     return loss


def find_loss(logits, actions, returns):
    y_true = actions
    y_pred = tf.math.argmax(logits, axis=-1)

    likelihood = tf.keras.losses.mean_squared_error(y_true, y_pred)

    return likelihood


def rolling_mean(x, n):
    return np.convolve(x, np.ones(n), "valid") / n


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