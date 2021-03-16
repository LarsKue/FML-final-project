
from .agents import DQBase, DQTrainer
from .utils import timer

import numpy as np
import os


def setup(self):

    use_existing_model = False
    self.model_path = "model"

    if use_existing_model:
        print("Using Existing Model:", self.model_path)

        if self.train:
            self.agent = DQTrainer.load(self.model_path)
        else:
            self.agent = DQBase.load(self.model_path)
    else:
        print("Creating Fresh Model from Scratch")
        self.model_path = self.model_path + "_fresh"
        actions = np.array(["UP", "DOWN", "LEFT", "RIGHT", "BOMB", "WAIT"])
        peaceful_actions = np.array(["UP", "DOWN", "LEFT", "RIGHT", "WAIT"])
        self.agent = DQTrainer(actions=actions, epsilon=1.0)


def act(self, game_state: dict):

    action = self.agent.act(game_state)

    return action
