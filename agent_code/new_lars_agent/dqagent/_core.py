
import tensorflow as tf
import numpy as np


class DQBase:
    def __init__(self, actions, model):
        self.actions = actions
        self.model = model

    def act(self, state: dict) -> str:
        action = self.exploit(state)
        return self.actions[action]

    def exploit(self, state: dict) -> int:
        """
        Exploit the current policy by choosing the best actions
        """
        observation = DQBase.observation(state)

        logits = self.model.predict(observation)

        return int(tf.math.argmax(logits, axis=-1))

    from ._io import load, save
    load = staticmethod(load)

    from ._observation import observation
