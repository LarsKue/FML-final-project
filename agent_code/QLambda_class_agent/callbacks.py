import numpy as np
from .QLambdaAgent import QLambdaAgent


def setup(self):
    weights = np.array([1.172, 0.602, -1.73, 0.324, -1.432, 0.823, -2.018, 0.885, -1.129, 0.141, 0.087, 0.107, 0.5, -0.5], dtype=float).T
    self.agent = QLambdaAgent(train=False, weights=weights)


def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)
