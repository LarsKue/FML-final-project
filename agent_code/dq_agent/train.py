from typing import List
import tensorflow as tf
import numpy as np
from shutil import rmtree
import json
from pathlib import Path

from .dqagent import DQTrain
from .utils import timer


def setup_training(self):

    self.actions = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])

    # initialize with weights from the coins task
    model_path = self.path.absolute().parents[0] / Path("test_3") / Path("model")
    model = tf.keras.models.load_model(model_path)

    if self.use_existing_model:
        print("Loading existing Agent...")
        self.agent = DQTrain.load(self.path)
    else:
        if self.path.is_dir():
            if self.overwrite:
                print("Deleting old Agent...")
                rmtree(self.path)
            else:
                raise FileExistsError(f"Agent '{self.path}' already exists.")

        if not self.path.is_dir():
            self.path.mkdir()

        e = self.settings["train"]["epsilon"]
        g = self.settings["train"]["gamma"]
        lr = self.settings["train"]["learning_rate"]

        # self.agent = DQTrain(self.actions, epsilon=e, gamma=g, learning_rate=lr)
        self.agent = DQTrain(self.actions, model=model, epsilon=e, gamma=g, learning_rate=lr)

        # copy the settings to the agent directory for later reference
        with open(self.path / Path("settings.json"), "w") as f:
            json.dump(self.settings, f)

    print("AGENT TYPE:", type(self.agent))
    print("AGENT EPSILON:", self.agent.epsilon)

    self.round_actions = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        # new episode
        self.agent.diagnostics.add_episode()
        self.agent.memory.add_episode()
        return

    self.round_actions += 1

    self.agent.memorize(old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    r = last_game_state["round"]

    self.agent.memorize(last_game_state, last_action, None, events)

    self.agent.finalize()
    if r % 50 == 0:
        print("Saving Agent...")
        self.agent.save(self.path)
        print("Saved.")
        print("Saving Diagnostics...")
        self.agent.diagnostics.save(self.path, r)
        print("Saved.")

    batch_size = 1024
    epochs = 5
    reduce = 50_000

    if batch_size * epochs <= len(self.agent.memory):
        with timer("Training took {:.2f} ms", transform=lambda d: 1000 * d, stdout=self.logger.debug):
            self.agent.train(epochs=epochs, batch_size=batch_size, reduce=reduce)
            self.agent.epsilon *= 0.999
            self.agent.epsilon = max(self.agent.epsilon, 0.1)

    self.round_actions = 0





