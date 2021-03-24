from typing import Tuple

import numpy as np
import tensorflow as tf
import _pickle as cpickle

import settings as s

from pathlib import Path

# tensorflow gpu has some issues on my machine
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
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

    @staticmethod
    def load(path):
        # saving or loading agents is non-trivial since tf.keras.model is only weakly referenced
        # and thus cannot be directly dumped to a file

        with open(path / Path("agent"), "rb") as f:
            instance = cpickle.load(f)

        instance.model = tf.keras.models.load_model(path / Path("model"))

        return instance

    def save(self, path):
        self.model.save(path / Path("model"))

        # temporarily remove the model, save the rest of the class and readd the model
        temp = self.__dict__.pop("model")
        with open(path / Path("agent"), "wb+") as f:
            cpickle.dump(self, f)

        self.model = temp

    def exploit(self, state: dict) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Exploit the current policy by choosing only the best actions
        """
        observation = self.observation(state)

        # predict the log probability of each action
        logits, prediction = self.predict(observation)

        # choose action according to probability distribution
        # this is equivalent to
        # action = np.random.choice(np.arange(len(self.actions)), p=prediction)
        # action = int(tf.random.categorical(logits, num_samples=1))

        action = np.argmax(prediction)

        return observation, prediction, action

    def predict(self, observation: np.ndarray):
        logits = self.model.predict(observation)
        prediction = tf.squeeze(tf.nn.softmax(logits, axis=-1)).numpy()

        return logits, prediction

    def act(self, state: dict) -> str:
        _, _, action = self.exploit(state)

        return self.actions[action]

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
            bombs[b[0][0], b[0][1]] = (5 - b[1]) / 4

        for c in state["coins"]:
            coins[c[0], c[1]] = 1.0

        player = np.zeros((s.COLS, s.ROWS), dtype=float)

        p = state["self"][3]

        player[p[0], p[1]] = 1.0 if state["self"][2] else -1

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

        return np.expand_dims(observation, axis=0)

    # def observation(self, state: dict) -> np.ndarray:
    #
    #     base_name = "_observation_"
    #     observation = np.array([])
    #
    #     i = 1
    #
    #     while True:
    #         nm = base_name + str(i)
    #
    #         if not hasattr(self, nm):
    #             break
    #
    #         o = getattr(self, base_name + str(i))(state)
    #
    #         observation = np.append(observation, o)
    #
    #         i += 1
    #
    #     return np.expand_dims(observation, axis=0)

    # def _observation_1(self, state: dict):
    #     # own position
    #     return np.array(state["self"][3])
    #
    # def _observation_2(self, state: dict):
    #     # position of other agents
    #
    #     result = np.zeros((3, 2))
    #
    #     for i, agent in enumerate(state["others"]):
    #         pos = np.array(agent[3])
    #         result[i] = pos
    #
    #     return result
    #
    # def _observation_3(self, state: dict):
    #     # 5x5 field around the agent
    #
    #     delta = 2
    #
    #     result = np.zeros((2 * delta + 1, 2 * delta + 1))
    #
    #     xpos, ypos = state["self"][3]
    #
    #     xmin = xpos - delta
    #     xmax = xpos + delta
    #     ymin = ypos - delta
    #     ymax = ypos + delta
    #
    #     for i, x in enumerate(range(xmin, xmax + 1)):
    #         if x < 0 or x >= s.COLS:
    #             continue
    #
    #         for j, y in enumerate(range(ymin, ymax + 1)):
    #             if y < 0 or y >= s.ROWS:
    #                 continue
    #
    #             result[i, j] = state["field"][x, y]
    #
    #     return result
    #
    # def __observation_4(self, state: dict):
    #     # coin locations
    #     result = np.zeros((9, 2))
    #
    #     for i, coin in enumerate(state["coins"]):
    #         result[i] = np.array(coin)
    #
    #     return result
    #
    # def _observation_5(self, state: dict):
    #     # bomb locations and timers
    #
    #     result = np.zeros((4, 3))
    #
    #     for i, b in enumerate(state["bombs"]):
    #         # x, y, t
    #         bomb = np.array([b[0][0], b[0][1], b[1]])
    #         result[i] = bomb
    #
    #     return result




