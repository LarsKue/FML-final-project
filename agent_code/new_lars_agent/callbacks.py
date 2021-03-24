import tensorflow as tf
from pathlib import Path
import numpy as np

from .dqagent import *


# tensorflow gpu has some issues on my machine
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def setup(self):
    self.use_existing_model = False
    self.path = Path("coins")
    self.overwrite = True

    if self.train:
        self.agent = None  # will be defined in setup_training
    else:
        if not self.use_existing_model:
            raise RuntimeError("Cannot create new agent in test-mode.")
        print("Loading existing agent...")
        self.agent = DQBase.load(self.path)


def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)
