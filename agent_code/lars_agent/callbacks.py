
from .agents import QAgent
from .utils import timer

import numpy as np

def setup(self):
    actions = np.array(["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"])

    peaceful_actions = np.array(["UP", "DOWN", "LEFT", "RIGHT", "WAIT"])

    self.agent = QAgent(actions=peaceful_actions, train=True)


def act(self, game_state: dict):

    with timer("Acting took {:.2f} ms", transform=lambda d: d * 1000):
        action = self.agent.act(game_state)

    return action
