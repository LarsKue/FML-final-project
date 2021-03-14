import numpy as np

from .features import feature_matrix
from .train import act_train



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    #self.weights = np.array([3.4, 3.7, -131.8, -82.6, -368.4, -21.4, -8.8, 8.8, -67.3], dtype=float).T
    #self.weights = np.array([0.8, 1.5, -21.4, 10.2, -40.8, 2.8, -3.2, 16.6, -28.9], dtype=float).T
    #self.weights = np.array([96.4, -24.7, -14.1, 48.9, -125.5, 31.8, -25.6, 25.8, -32.0], dtype=float).T
    #self.weights = np.array([60.5, -5.4, -38.2, 121.0, 8.9, 29.7, -25.7, 16.4, -10.8], dtype=float).T
    #self.weights = np.array([11.39, 1.98, -5.64, 4.08, -5.54, 9.45, -6.27, 3.81, -0.32, 3,1,1], dtype=float).T
    #self.weights = np.array([1.627, 1.995, -0.774, 1.392, -2.417, 0.717, -0.897, 1.734, -1.033, 0.542, 0.985, 0.987], dtype=float).T
    self.weights = np.array([2.375, 1.059, -1.087, 0.569, -0.496, 0.805, 0.175, 0.845, -0.028, 0.077, 0.039, 0.333], dtype=float).T
    self.weights = np.array([1.172, 0.602, -1.73, 0.324, -1.432, 0.823, -0.018, 0.885, -1.129, 0.141, 0.087, 0.107], dtype=float).T


    self.previous_action = None




def act(self, game_state: dict) -> str:
    if self.train:
        return act_train(self, game_state)
    else:
        
        features = feature_matrix(self, game_state)

        q = self.weights.dot(features)


        best_actions = np.argwhere(q == np.amax(q)).flatten()

        action = ACTIONS[np.random.choice(best_actions)]
        self.previous_action =  action

        """
        print("\n")
        print(np.round(self.weights, 3))
        print(features)
        print(np.round(q, 3))
        print(action)
        """
        
        return action