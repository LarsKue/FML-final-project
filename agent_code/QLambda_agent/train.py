from typing import List
import numpy as np
from collections import deque

import events as e

from .QLambdaAgent import QLambdaAgent


def setup_training(self):
    self.evaluate = False
    if self.evaluate:
        self.rounds_per_weight = 10
        self.weights = np.load('weights_updates.npy')
        self.num_weights = np.where(~self.weights[1:].any(axis=1))[0]
        self.num_weights = len(self.weights) if len(self.num_weights) == 0 else self.num_weights[0]
        print("\n", self.num_weights)
        self.weights = self.weights[:self.num_weights]

        self.coins_collected = np.zeros((self.num_weights, self.rounds_per_weight))
        self.kills = np.zeros((self.num_weights, self.rounds_per_weight))
        self.got_killed = np.zeros((self.num_weights, self.rounds_per_weight))
        self.suicides = np.zeros((self.num_weights, self.rounds_per_weight))
        self.invalid_action = np.zeros((self.num_weights, self.rounds_per_weight))
        self.rewards = np.zeros((self.num_weights, self.rounds_per_weight))
        self.survived_round = np.zeros((self.num_weights, self.rounds_per_weight))
        self.steps = np.zeros((self.num_weights, self.rounds_per_weight))
        

        self.current_weight_index = 0
        self.current_round_number = 0
        

        with np.printoptions(precision=3, suppress=True, linewidth=200):
            print(self.weights[0])

        self.agent = QLambdaAgent(train=False, weights=self.weights[0])
    else:
        self.agent = QLambdaAgent(train=True)

        self.rounds_per_weight = 10
        self.num_weights = 2001

        self.coins_collected = np.zeros((self.num_weights, self.rounds_per_weight))
        self.kills = np.zeros((self.num_weights, self.rounds_per_weight))
        self.got_killed = np.zeros((self.num_weights, self.rounds_per_weight))
        self.suicides = np.zeros((self.num_weights, self.rounds_per_weight))
        self.invalid_action = np.zeros((self.num_weights, self.rounds_per_weight))
        self.rewards = np.zeros((self.num_weights, self.rounds_per_weight))
        self.survived_round = np.zeros((self.num_weights, self.rounds_per_weight))
        self.steps = np.zeros((self.num_weights, self.rounds_per_weight))

        self.current_weight_index = 0
        self.current_round_number = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        return

    if self.evaluate:
        self.coins_collected[self.current_weight_index, self.current_round_number] += events.count(e.COIN_COLLECTED)
        self.kills[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_OPPONENT)
        self.got_killed[self.current_weight_index, self.current_round_number] += events.count(e.GOT_KILLED)
        self.suicides[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_SELF)
        self.invalid_action[self.current_weight_index, self.current_round_number] += events.count(e.INVALID_ACTION)
        self.rewards[self.current_weight_index, self.current_round_number] += self.agent._calculate_reward(events)
    
    else:
        self.agent.add_sars(old_game_state, self_action, new_game_state, events)
        
        self.coins_collected[self.current_weight_index, self.current_round_number] += events.count(e.COIN_COLLECTED)
        self.kills[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_OPPONENT)
        self.got_killed[self.current_weight_index, self.current_round_number] += events.count(e.GOT_KILLED)
        self.suicides[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_SELF)
        self.invalid_action[self.current_weight_index, self.current_round_number] += events.count(e.INVALID_ACTION)
        self.rewards[self.current_weight_index, self.current_round_number] += self.agent._calculate_reward(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    if self.evaluate:
        self.coins_collected[self.current_weight_index, self.current_round_number] += events.count(e.COIN_COLLECTED)
        self.kills[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_OPPONENT)
        self.got_killed[self.current_weight_index, self.current_round_number] += events.count(e.GOT_KILLED)
        self.suicides[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_SELF)
        self.invalid_action[self.current_weight_index, self.current_round_number] += events.count(e.INVALID_ACTION)
        self.rewards[self.current_weight_index, self.current_round_number] += self.agent._calculate_reward(events)

        self.survived_round[self.current_weight_index, self.current_round_number] = events.count(e.SURVIVED_ROUND)
        self.steps[self.current_weight_index, self.current_round_number] = last_game_state['step']


        self.current_round_number += 1

        if self.current_round_number == self.rounds_per_weight:
            self.current_round_number = 0
            self.current_weight_index += 1

            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print(np.round(self.weights[self.current_weight_index], 3))
            self.agent = QLambdaAgent(train=False, weights=self.weights[self.current_weight_index])

            if self.current_weight_index > 0 and self.current_weight_index % 10 == 0:
                data = np.stack([self.coins_collected,
                                self.kills,
                                self.got_killed,
                                self.suicides,
                                self.invalid_action,
                                self.rewards,
                                self.survived_round,
                                self.steps])

                np.save('evaluation_data.npy', data)

    else:
        self.coins_collected[self.current_weight_index, self.current_round_number] += events.count(e.COIN_COLLECTED)
        self.kills[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_OPPONENT)
        self.got_killed[self.current_weight_index, self.current_round_number] += events.count(e.GOT_KILLED)
        self.suicides[self.current_weight_index, self.current_round_number] += events.count(e.KILLED_SELF)
        self.invalid_action[self.current_weight_index, self.current_round_number] += events.count(e.INVALID_ACTION)
        self.rewards[self.current_weight_index, self.current_round_number] += self.agent._calculate_reward(events)

        self.survived_round[self.current_weight_index, self.current_round_number] = events.count(e.SURVIVED_ROUND)
        self.steps[self.current_weight_index, self.current_round_number] = last_game_state['step']

        self.current_round_number += 1

        self.agent.add_sars(None, last_action, last_game_state, events)

        if last_game_state['round'] > 0 and last_game_state['round'] % 10 == 0:
            self.agent.update_weights()

            self.agent.print_weights()
            
            
            self.current_round_number = 0
            self.current_weight_index += 1

            data = np.stack([self.coins_collected,
                            self.kills,
                            self.got_killed,
                            self.suicides,
                            self.invalid_action,
                            self.rewards,
                            self.survived_round,
                            self.steps])

            np.save(f"training_evaluation_data_{self.agent.name}.npy", data)
