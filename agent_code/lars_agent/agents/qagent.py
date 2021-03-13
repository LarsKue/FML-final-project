
import numpy as np
from numba import jit, cuda

from ..utils import *


class QAgent:
    def __init__(self, actions: np.ndarray, epsilon = 1.0, train = True, temperature = 1.0):
        self.actions = actions
        self.epsilon = epsilon
        self.train = train

        self.feature_action_counts = {}
        self.temperature = temperature

    def act(self, state: dict):
        if self.train and rand(self.epsilon):
            return self.explore(state)

        return self.exploit(state)

    def explore(self, state: dict):
        features = tuple(self.features(state))

        action_counts = self.feature_action_counts.get(features)

        if action_counts is None:
            action_counts = np.zeros(self.actions.shape, dtype=int)
            self.feature_action_counts[features] = action_counts

        # prefer actions we have not tried much, randomized for ties
        values = softmin(action_counts / self.temperature)

        best_actions = np.squeeze(np.argwhere(values == np.max(values)))

        if best_actions.ndim == 0:
            action = best_actions
        else:
            action = np.random.choice(best_actions)

        # increment action counter
        action_counts[action] += 1

        return self.actions[action]

    def exploit(self, state: dict):
        # transform the game state into features
        features = self.features(state)
        # transform the features into action values
        values = self.values(features)

        # return the best action, randomized for ties
        best_actions = np.squeeze(np.argwhere(values == np.max(values)))

        if best_actions.ndim == 0:
            action = best_actions
        else:
            action = np.random.choice(best_actions)

        return self.actions[action]

    def features(self, state: dict):
        """
        Transform Game State into Features
        """
        features = np.array([1])

        return features

    def values(self, features):
        """
        Transform features into values
        """
        values = np.ones(self.actions.shape)

        return values

    def reward(self):
        pass

