import tensorflow as tf
import numpy as np
from pathlib import Path
from .dqagent import *

import os

# ensure CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def setup(self):
    actions = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])
    model = tf.keras.models.load_model(Path("model"))

    self.agent = DQBase(actions, model)


def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)
