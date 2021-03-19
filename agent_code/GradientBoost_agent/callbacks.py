import numpy as np
import pickle

from .features import get_features
from .train import act_train



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