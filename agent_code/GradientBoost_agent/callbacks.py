import numpy as np
import pickle

from .features import get_features
from .train import act_train

import settings as s



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    with open('forest_data', 'rb') as f:
        self.model = pickle.load(f)

    return



def act(self, game_state: dict) -> str:
    if self.train:
        return act_train(self, game_state)
    else:

        features = get_features(self, game_state)
        
        #print("")
        #print(features)
        #print(self.model.predict([features])[0])

        q = self.model.predict([features])[0]

        best_actions = np.argwhere(q == np.amax(q)).flatten()

        a = ACTIONS[np.random.choice(best_actions)]

        return a
        
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
