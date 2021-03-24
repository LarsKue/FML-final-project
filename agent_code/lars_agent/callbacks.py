
from .agents import DQBase, DQTrainer
from .utils import timer

import numpy as np
from pathlib import Path
import shutil


def setup(self):

    use_existing_model = True
    self.model_path = Path("coins_test")
    overwrite = False

    if use_existing_model:
        print("Using Existing Model:", self.model_path)

        if self.train:
            self.agent = DQTrainer.load(self.model_path)
        else:
            self.agent = DQBase.load(self.model_path)
    else:
        print("Creating Fresh Model from Scratch")
        if self.model_path.is_dir():
            if overwrite:
                # overwrite if the model already exists
                print("Deleting previous model...")
                shutil.rmtree(self.model_path)
            else:
                raise FileExistsError("Model already exists. Set overwrite = True to ignore this error.")

        actions = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])
        peaceful_actions = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT"])
        coins_actions = np.array(["UP", "RIGHT", "DOWN", "LEFT"])
        self.agent = DQTrainer(actions=actions)

    if not self.train:
        self.agent.epsilon = 0

    self.figures_path = self.model_path / Path("figures")
    self.data_path = self.model_path / Path("data")

    self.model_path.mkdir(exist_ok=True)
    self.figures_path.mkdir(exist_ok=True)
    self.data_path.mkdir(exist_ok=True)


def act(self, game_state: dict):

    action, prediction = self.agent.act(game_state)

    if self.train:
        self.predictions.append(prediction)
        self.last_episode_predictions.append(prediction)

    return action
