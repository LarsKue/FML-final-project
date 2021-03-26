import numpy as np
from .QLambdaAgent import QLambdaAgent


def setup(self):
    weights = np.array([1.172, 0.602, -1.73, 0.324, -1.432, 0.823, -2.018, 0.885, -1.129, 0.141, 0.087, 0.107, 0.5, -0.5], dtype=float).T
    weights = np.array([1.252, 0.557, -1.07, 0.322, -1.063, 0.722, -0.899, 0.844, -0.667, 0.235, 0.01, 0.039, 0.071, -1.492])
    weights = np.array([1.313, 0.504, -0.958, 0.317, -2.223, 0.737, -0.994, 0.866, -0.748, -0.255, 0.019, -0.001, 0.258, -2.228]).T
    weights = np.array([1.495, 0.484, -0.985, 0.528, -1.875, 1.076, -0.966, 1.276, 0.0, -1.186, -0.073, -0.013, 0.0, -0.403])
    self.agent = QLambdaAgent(train=False, weights=weights)


def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)
