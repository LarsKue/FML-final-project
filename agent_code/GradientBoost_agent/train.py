from typing import List
import numpy as np
from collections import deque

import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
#from lightgbm import LGBMRegressor

from .features import get_features
import events as e
import settings as s



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
    """
    features:
        - top
        - right
        - bottom
        - left
        - nearest coin distance
        - nearest coin x
        - nearest coin y
    """

    self.epsilon = 0.25
    self.epsilon_decrease = 0.99

    self.alpha = 0.1

    self.discout = 0.9
    
    
    self.action_indices = {
        'UP': 0,
        'RIGHT': 1,
        'DOWN': 2,
        'LEFT': 3,
        'WAIT': 4,
        'BOMB': 5,
    }

    self.experience_buffer = []
    self.batch_size = 5000

    #self.model = MultiOutputRegressor(LGBMRegressor(n_estimators=30))
    #self.model = RandomForestRegressor(n_estimators=30)
    self.model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, warm_start=True))


    self.is_fit = False



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state == None:
        return


    self.experience_buffer.append([
        observation(self, old_game_state),
        self_action,
        calculate_reward(events),
        observation(self, new_game_state),
        False
    ])




def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    self.experience_buffer.append([
        self.experience_buffer[-1][3],
        last_action,
        calculate_reward(events),
        observation(self, last_game_state),
        True
    ])

    
    
    if last_game_state['round'] < 10 or last_game_state['round'] % 50 == 0:
        experience_replay(self)
        self.batch_size += 2000
        self.batch_size = min(self.batch_size, 20000)

        self.epsilon *= self.epsilon_decrease
        self.epsilon = max(self.epsilon, 0.1)

        with open('forest_data', 'wb') as f:
            pickle.dump(self.model, f)

        
        print("\n", self.epsilon, self.alpha)


    if last_game_state['round'] > 0 and last_game_state['round'] % 1000 == 0:
        self.experience_buffer = []

    
    

def calculate_reward(events: List[str]):
        reward = 0

        for event in events:
            if event == e.INVALID_ACTION:
                reward += -200
            elif event == e.CRATE_DESTROYED:
                reward += 10
            elif event == e.COIN_FOUND:
                reward += 30
            elif event == e.COIN_COLLECTED:
                reward += 150
            elif event == e.KILLED_OPPONENT:
                reward += 500
            elif event == e.KILLED_SELF:
                reward += 0              # sd == KILLED_SELF + GOT_KILLED => reward(sd) = -1250
            elif event == e.GOT_KILLED:
                reward += -1250
            else:
                reward -= 5

        return reward / 100


def epsilon_greedy(self, game_state: dict):

    if not self.is_fit or np.random.uniform() < self.epsilon:
        return np.random.choice(ACTIONS)

    else:
        
        features = observation(self, game_state)

        q = self.model.predict([features])[0]

        best_actions = np.argwhere(q == np.amax(q)).flatten()

        a = ACTIONS[np.random.choice(best_actions)]

        return a



def act_train(self, game_state: dict):
    action = epsilon_greedy(self, game_state)
    return action



def experience_replay(self):
    print("\nupdate")

    batch_size = min(self.batch_size, len(self.experience_buffer))
    batch_indices = np.random.choice(np.arange(len(self.experience_buffer)), size=batch_size, replace=False)


    X = []
    targets = []
    for i in batch_indices:
        old_state, action, reward, new_state, is_terminal = self.experience_buffer[i]

        Y_i = reward
        if not is_terminal and self.is_fit:
            #print(action, reward, self.model.predict([new_state])[0])
            Y_i += self.discout * np.max(self.model.predict([new_state])[0])
        
        if self.is_fit:
            q_values = self.model.predict([old_state])
        else:
            q_values = np.zeros(len(ACTIONS)).reshape(1, -1)

        
        #print("\n")
        #print(q_values)
        q_values[0][self.action_indices[action]] += self.alpha * (Y_i - q_values[0][self.action_indices[action]])
        #print(q_values)
        
        X.append(old_state)
        targets.append(q_values[0])

    

    #print(np.array(targets))

    self.model.fit(X, targets)
    self.is_fit = True



def observation(self, state: dict) -> np.ndarray:
    # different approach:
    # like with chess, transform the state into
    # an M x N x K board with each K-layer
    # being a flag for what is there:
    # walls
    # crates
    # bombs
    # coins
    # the player
    # enemies

    # walls and crates are given by the 'field' in the state
    walls = (state["field"] == -1).astype(float)
    crates = (state["field"] == 1).astype(float)

    # bombs and coins are individual positions
    bombs = np.zeros((s.COLS, s.ROWS), dtype=float)
    coins = np.zeros((s.COLS, s.ROWS), dtype=float)

    for b in state["bombs"]:
        bombs[b[0][0], b[0][1]] = (5. - b[1]) / 4.

    for c in state["coins"]:
        coins[c[0], c[1]] = 1.0

    player = np.zeros((s.COLS, s.ROWS), dtype=float)

    p = state["self"][3]

    player[p[0], p[1]] = 1.0 if state["self"][2] else -1.0

    enemies = np.zeros((s.COLS, s.ROWS), dtype=float)

    for o in state["others"]:
        op = o[3]
        enemies[op[0], op[1]] = 1.0

    observation = np.stack([
        walls,
        crates,
        bombs,
        coins,
        player,
        enemies
    ], axis=-1)

    return np.expand_dims(observation, axis=0).flatten()
