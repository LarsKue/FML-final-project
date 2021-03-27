
import unittest

import numpy as np

import utils


class UtilsTest(unittest.TestCase):

    def test_discount(self):
        rewards = np.array([5, 10, 15, 20, 30, 50])
        gamma = 0.9

        discounted_rewards = np.array([5, 9, 12.15, 14.58, 19.683, 29.5245])

        self.assertTrue(np.allclose(discounted_rewards, utils.discount(rewards, gamma)))

    def test_returns(self):
        rewards = np.array([5, 10, 15, 20, 30, 50])
        gamma = 0.9

        returns = np.array([89.9375, 84.9375, 75.9375, 63.7875, 49.2075, 29.5245])

        self.assertTrue(np.allclose(returns, utils.returns(rewards, gamma)))


if __name__ == "__main__":
    unittest.main()
