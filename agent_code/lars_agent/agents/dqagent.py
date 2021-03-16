import numpy as np
import tensorflow as tf
import _pickle as cpickle

from ..utils import rand, discount, find_loss, softmin
from .memory import Memory
import events as e


# tensorflow gpu has some issues on my machine
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DQBase:
    def __init__(self, actions: np.ndarray, model: tf.keras.Model):
        self.actions = actions
        self.model = model

    def __new__(cls, **fields):
        # instantiate with arbitrary fields
        obj = super().__new__(cls)
        for (k, v) in fields.items():
            setattr(obj, k, v)

        return obj

    @classmethod
    def load(cls, path):
        # saving or loading agents is non-trivial since tf.keras.model is only weakly referenced
        # and thus cannot be directly dumped to a file
        with open(path + ".agent", "r") as f:
            d = cpickle.load(f)

        result = cls.__new__(**d)
        result.model = tf.keras.models.load_model(path + ".model")

    def save(self, path):
        self.model.save(path + ".model")

        # temporarily remove the model, save the rest of the class and readd the model
        temp = self.__dict__.pop("model")
        with open(path + ".agent", "wb+") as f:
            cpickle.dump(self, f)

        self.model = temp

    def exploit(self, state: dict) -> [np.ndarray, int]:
        """
        Exploit the current policy by choosing only the best actions
        """
        observation = self.observation(state)
        # predict the log probability of each action
        logits = self.model.predict(observation)

        # return the best action, randomized for ties
        best_actions = np.squeeze(np.argwhere(logits == np.max(logits)))

        if best_actions.ndim == 0:
            action = best_actions
        else:
            action = np.random.choice(best_actions)

        return observation, action

    def act(self, state: dict) -> str:
        _, action = self.exploit(state)

        return self.actions[action]

    def observation(self, state: dict) -> np.ndarray:
        observation = state["field"].flatten()
        return np.expand_dims(observation, axis=0)


class DQTrainer(DQBase):
    def __init__(self, actions: np.ndarray, epsilon=1.0, temperature=1.0, gamma=0.9):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=32, activation="sigmoid"),  # hidden layer
            tf.keras.layers.Dense(units=len(actions), activation=None)  # output layer
        ])

        self.epsilon = epsilon
        self.temperature = temperature
        self.discount = gamma

        self.memory = Memory()

        self.feature_action_counts = {}

        super().__init__(actions, model)

    def to_base(self) -> DQBase:
        return DQBase(self.actions, self.model)

    def act(self, state: dict) -> str:
        if rand(self.epsilon):
            observation, action = self.explore(state)
        else:
            observation, action = super().exploit(state)

        # memorize what we did
        self.memory.add_observation(observation)
        self.memory.add_action(action)

        return self.actions[action]

    def explore(self, state: dict) -> [np.ndarray, int]:
        """
        Explore the game world outside of the current policy
        """
        observation = super().observation(state)

        # hashable observational state as dict keys
        h = tuple(np.squeeze(observation))

        action_counts = self.feature_action_counts.get(h)

        if action_counts is None:
            # have not been in this state yet
            action_counts = np.zeros(self.actions.shape, dtype=int)
            self.feature_action_counts[h] = action_counts

        # prefer actions we have not tried much, randomized for ties
        values = softmin(action_counts / self.temperature)

        best_actions = np.squeeze(np.argwhere(values == np.max(values)))

        if best_actions.ndim == 0:
            action = best_actions
        else:
            action = np.random.choice(best_actions)

        # increment action counter
        action_counts[action] += 1

        return observation, action

    def train(self, optimizer):
        discounted_rewards = discount(self.memory.rewards, self.discount)

        with tf.GradientTape() as tape:
            # Forward Propagation
            logits = self.model(np.vstack(self.memory.observations))

            loss = find_loss(logits, np.array(self.memory.actions), discounted_rewards)

        # Back Propagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if self.epsilon > 0.2:
            self.epsilon *= 0.95

        return loss, np.sum(discounted_rewards)

    def reward(self, events):
        event_rewards = {
            e.INVALID_ACTION: -200,
            e.CRATE_DESTROYED: 10,
            e.COIN_FOUND: 30,
            e.COIN_COLLECTED: 500,
            e.KILLED_OPPONENT: 300,
            e.KILLED_SELF: -300,
            e.GOT_KILLED: -700,
        }

        rewards = list(map(event_rewards.get, events))
        rewards = [r if r is not None else -1 for r in rewards]
        reward = np.sum(rewards)

        # memorize the reward
        self.memory.add_reward(reward)

        return reward

