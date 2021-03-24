from typing import List
import tensorflow as tf
import numpy as np
from shutil import rmtree

from .dqagent import DQTrain


def setup_training(self):

    self.actions = np.array(["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"])

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

        self.agent = DQTrain(self.actions)

    self.round_actions = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    self.round_actions += 1

    self.agent.memorize(old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    r = last_game_state["round"]
    print(f"Completed round {r} after {self.round_actions} actions.")

    self.agent.memorize(last_game_state, last_action, None, events)

    self.agent.finalize()
    if r % 30 == 0:
        # self.agent.save_diagnostics(self.path, r)
        print("Saving Agent...")
        self.agent.save(self.path)
        print("Saved.")
        print("Saving Diagnostics...")
        self.agent.diagnostics.save(self.path, r)
        print("Saved.")

    batch_size = 512
    epochs = 3
    reduce = 20_000

    if batch_size * epochs >= len(self.agent.memory):
        self.agent.train(epochs=epochs, batch_size=batch_size, reduce=reduce)

    self.round_actions = 0





