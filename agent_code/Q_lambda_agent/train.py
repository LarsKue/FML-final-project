from typing import List
import numpy as np
from collections import deque

from .features import feature_matrix, feature_vector
import events as e



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    self.discount = 0.95
    self.lambda_et = 0.95

    self.alpha = 0.001
    self.alpha_decrease = 1#0.999

    self.epsilon = 0.15
    self.epsilon_decrease = 0.99
    self.min_epsilon = 0.10

    self.states_count = {}
    self.exploration_parameter = 10
    self.total_state_count = 0
    
    self.action_indices = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3,
        'WAIT': 4,
        'BOMB': 5,
    }



    self.weights = np.random.uniform(-1, 1, 12).T
    self.weights = np.array([2, 2, -1, 3, -3, 1, -1, 3, -1, 1, 1, 1], dtype=float).T
    self.weights = np.array([1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1], dtype=float).T
    self.weights = np.zeros(12)


    self.previous_action = None

    self.action_was_random = False
    
    
    self.experience_buffer = [[]]
    self.batch_size = 500

    self.weights_updates = np.zeros((2001, 12))
    self.update_number = 0
    
    

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state == None:
        return

    self.experience_buffer[-1].append([
        feature_matrix(self, old_game_state), 
        self_action, 
        calculate_reward(events),
        feature_matrix(self, new_game_state),
        self.action_was_random
    ])



    self.previous_action = self_action
    



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):

    self.experience_buffer[-1].append([
        self.experience_buffer[-1][-1][3], 
        last_action, 
        calculate_reward(events), 
        feature_matrix(self, last_game_state),
        self.action_was_random
    ])



    if last_game_state['round'] > 0 and last_game_state['round'] % 10 == 0:
        Watkins_Q_lambda(self)
        self.experience_buffer = [[]]
        
        #self.batch_size += 100
        #self.batch_size = min(self.batch_size, 10000)

        self.update_number += 1
        self.weights_updates[self.update_number] = self.weights

        #self.alpha *= self.alpha_decrease
        #self.alpha = max(self.alpha, 0.001)

        #n = self.update_number
        #self.alpha = 1/(0.3*n+75)


    if last_game_state['round'] > 0 and last_game_state['round'] % 50 == 0:
        self.epsilon *= self.epsilon_decrease
        self.epsilon = max(self.epsilon, 0.05)




    if last_game_state['round'] > 0 and last_game_state['round'] % 200 == 0:
        np.save('weights_updates.npy', self.weights_updates)



    
    if last_game_state['round'] > 0 and last_game_state['round'] % 10 == 0:
        print("")
        for w in self.weights:
            print(f"{w:0.3f}, ", sep="", end="")
            
        #print("\n")
        #for state, count in self.states_count.items():
        #    print(f"{state}:\t{count}")
        print("\n", self.epsilon, self.alpha)



def calculate_reward(events: List[str]):
    reward = 0

    for event in events:
        if event == e.INVALID_ACTION:
            reward += -200
        elif event == e.BOMB_DROPPED:
            reward += 5
        elif event == e.CRATE_DESTROYED:
            reward += 10
        elif event == e.COIN_FOUND:
            reward += 30
        elif event == e.COIN_COLLECTED:
            reward += 150
        elif event == e.KILLED_OPPONENT:
            reward += 500
        elif event == e.KILLED_SELF:
            reward += -300              # sd == KILLED_SELF + GOT_KILLED => reward(sd) = -1000
        elif event == e.GOT_KILLED:
            reward += -700
        #elif event == e.SURVIVED_ROUND:
        #    reward += 20

    return reward / 100


def epsilon_greedy(self, game_state: dict):
    features = feature_matrix(self, game_state)

    epsilon = self.epsilon


    """
    state_string = np.array2string(features.flatten(), separator='', formatter={'float_kind': lambda x: f"{x:0.0f}"})[1:-2]
    if state_string in self.states_count:
        epsilon *= np.exp(-self.exploration_parameter * self.states_count[state_string] / self.total_state_count)
        self.states_count[state_string] += 1
    else:
        self.states_count[state_string] = 1
    epsilon += self.min_epsilon
    self.total_state_count += 1
    #print("\n")
    """

    if np.random.uniform() < epsilon:
        self.action_was_random = True
        a = np.random.choice(ACTIONS)
        return a
    else:
        self.action_was_random = False


        q = self.weights.dot(features)

        best_actions = np.argwhere(q == np.amax(q)).flatten()

        a = ACTIONS[np.random.choice(best_actions)]

        return a
        



def act_train(self, game_state: dict):
    action = epsilon_greedy(self, game_state)
    return action




def Watkins_Q_lambda(self):

    for episode in self.experience_buffer:
        eligibility_trace = np.zeros(len(self.weights))
        for old_state, action, reward, new_state, action_was_random in episode:
            eligibility_trace += old_state[:,self.action_indices[action]]


            delta = reward - self.weights.dot(old_state[:,self.action_indices[action]])

            Q = self.weights.dot(new_state)

            delta += self.discount * np.max(Q)

            self.weights += self.alpha * delta * eligibility_trace

            if action_was_random:
                eligibility_trace = np.zeros(len(self.weights))
            else:
                eligibility_trace *= self.discount * self.lambda_et
