import numpy as np
from .SorryDaveAgent import SorryDaveAgent


def setup(self):
    weights = np.array([1.388, 0.719, -0.927, 0.77, -1.606, 0.87, -1.561, 1.488, -1.735, -0.054, -0.05, -0.52])
    self.agent = SorryDaveAgent(train=False, weights=weights, logger=self.logger)


def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)
