
from ..utils import *

from collections import deque


class Memory:
    def __init__(self, observations=None, actions=None, rewards=None, returns=None):
        if observations is None:
            self.observations = []
        else:
            self.observations = list(observations)
        if actions is None:
            self.actions = []
        else:
            self.actions = list(actions)
        if rewards is None:
            self.rewards = []
        else:
            self.rewards = list(rewards)

        if returns is None:
            self.returns = []
        else:
            self.returns = list(returns)

        self.__complete = False
        self.__check_complete()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.returns = []

        self.__complete = True

    def add(self, observations, action, reward, ret):
        self.observations.append(observations)
        self.actions.append(action)
        self.rewards.append(reward)
        self.returns.append(ret)

    def add_observation(self, observation):
        self.observations.append(observation)
        self.__check_complete()

    def add_action(self, action):
        self.actions.append(action)
        self.__check_complete()

    def add_reward(self, reward):
        self.rewards.append(reward)
        self.__check_complete()

    def __check_complete(self):
        self.__complete = all_equal(len(self.observations), len(self.actions), len(self.rewards), len(self.returns))
        return self.__complete

    def random_batch(self, size, return_indices=False):
        if size >= len(self):
            if return_indices:
                return self, np.arange(len(self))
            else:
                return self

        idx = np.random.choice(np.arange(len(self)), size, replace=False)

        result = Memory(
                np.asarray(self.observations)[idx],
                np.asarray(self.actions)[idx],
                np.asarray(self.rewards)[idx],
                np.asarray(self.returns)[idx],
        )

        if return_indices:
            return result, idx
        else:
            return result

    def reduce(self, new_size):
        if new_size >= len(self):
            return

        self.observations = self.observations[-new_size:]
        self.actions = self.actions[-new_size:]
        self.rewards = self.rewards[-new_size:]
        self.returns = self.returns[-new_size:]

    def __len__(self):
        if not self.__check_complete():
            raise RuntimeError(f"Cannot define length on incomplete Memory: {(len(self.observations), len(self.actions), len(self.rewards), len(self.returns))}")
        return len(self.observations)
