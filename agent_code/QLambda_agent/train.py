from typing import List
import numpy as np
from collections import deque

import events as e

from .QLambdaAgent import QLambdaAgent


def setup_training(self):
    self.agent = QLambdaAgent(train=True)

    self.agent.print_weights()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    self.agent.add_sars(old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.agent.add_sars(None, last_action, last_game_state, events)

    if last_game_state['round'] > 0 and last_game_state['round'] % 10 == 0:
        self.agent.update_weights()

        self.agent.print_weights()
