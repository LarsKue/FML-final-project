import tensorflow as tf


class DQBase:
    def __init__(self, actions, model):
        self.actions = actions
        self.model = model

    def act(self, state: dict) -> str:
        action = self.exploit(state)
        return self.actions[action]

    def exploit(self, state: dict) -> int:
        observation = DQBase.observation(state)

        logits = self.model(observation)

        return tf.math.argmax(logits, axis=-1)

    from ._observation import observation
