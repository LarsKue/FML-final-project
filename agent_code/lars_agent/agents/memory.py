
from ..utils import *


class Memory:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []

        self.__complete = True

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

        self.__complete = True

    def add(self, observations, values, action, reward):
        self.observations.append(observations)
        self.actions.append(action)
        self.rewards.append(reward)

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
        self.__complete = all_equal(len(self.observations), len(self.actions), len(self.rewards))

    @staticmethod
    def aggregate(*memories):
        batch = Memory()

        for m in memories:
            for step in zip(m.observations, m.actions, m.rewards):
                batch.add(*step)

        return batch

    def __len__(self):
        if not self.__complete:
            raise RuntimeError("Cannot define length on incomplete Memory")
        return len(self.observations)
