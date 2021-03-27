import tensorflow as tf
from pathlib import Path
import numpy as np

import json

from .dqagent import *

# tensorflow gpu has some issues on my machine
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def setup(self):

    # find the index of the current agent to allow initializing multiple agents
    # and training them in the same instance (self-play)
    # base_name = "new_lars_agent_"
    # index = int(self.logger.name[len(base_name)])
    index = 0

    settings_path = Path("settings.json")
    with open(settings_path, "r") as f:
        self.settings = json.load(f)

    self.use_existing_model = self.settings["setup"]["use_existing_model"]
    self.path = Path(self.settings["setup"]["path"])
    self.path = self.path.with_name(self.path.name + "_" + str(index))
    self.overwrite = self.settings["setup"]["overwrite"]

    if self.train:
        self.agent = None  # will be defined in setup_training
    else:
        if not self.use_existing_model:
            raise RuntimeError("Cannot create new agent in test-mode.")
        print("Loading existing agent as DQBase...")
        agent = DQBase.load(self.path)
        # must downcast to base class to prevent random actions etc.
        self.agent = DQBase(agent.actions, agent.model)


def act(self, game_state: dict) -> str:
    return self.agent.act(game_state)
