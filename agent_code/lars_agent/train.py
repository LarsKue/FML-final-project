from typing import List
import tensorflow as tf
from scipy import interpolate
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import rolling_mean
from .test_states import test_states

import settings as s


"""

Order of Execution:

In Tournament:

setup()
act()
act()
...


In Training:

setup()
setup_training()
game_events_occurred()
act()
game_events_occurred()
act()
...

"""


def setup_training(self):
    self.learning_rate = 1e-4
    self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    self.rewards = []
    self.losses = []
    self.epsilons = []
    self.steps_taken = []
    self.predictions = []
    self.predictions = []

    # must define input shape here since this is very slow (~1.5s) and otherwise happens in the first call to act()
    # so pretend to have some state to observe and pass to model.predict()

    pretend_state = {
        "round": 1,
        "step": 15,
        "field": np.zeros((s.COLS, s.ROWS)),
        "self": ("lars_agent", 0, False, (0, 0)),
        "others": [
            ("rule_based_agent_0", 0, False, (0, 1)),
            ("rule_based_agent_1", 0, False, (1, 0)),
            ("rule_based_agent_2", 0, False, (1, 1))
        ],
        "bombs": [],
        "coins": [],
        "user_input": "WAIT",
        "explosion_map": np.zeros((s.COLS, s.ROWS))
    }

    observation = self.agent.observation(pretend_state)
    self.agent.model.predict(observation)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if old_game_state is None:
        # game has not started yet
        return

    self.agent.reward(old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    r = last_game_state["round"]

    self.agent.reward(last_game_state, last_action, None, events)

    gamma = 0.98
    batch_size = 32
    reduce = 1024  # MB

    reduce = 1e6 * reduce / (17 * 17 * 6 * 4)
    reduce = int(reduce)

    if batch_size > reduce:
        raise RuntimeError(f"Batch size cannot be greater than maximum memory size: {batch_size} vs {reduce}")

    steps = last_game_state["step"]

    def update_epsilon(e):
        min_e = 0.01
        sink_rate = 1e-1
        if e > min_e:
            return e * (1 / (1 + sink_rate * e))
        return min_e

    loss, reward = self.agent.train(self.optimizer, gamma, epsilon_updater=update_epsilon, batch_size=batch_size,
                                    reduce_memory=reduce, clear_memory=False)

    if loss is None:
        loss = np.nan

    self.rewards.append(reward)
    self.losses.append(loss)
    self.epsilons.append(self.agent.epsilon)
    self.steps_taken.append(steps)

    print("Finished round", r)
    if r % 30 == 0:
        print("Saving Agent...")
        # save the agent to disk
        self.agent.save(self.model_path)
        print("Saved!")

        rounds = np.arange(0, r)
        print("Saving reward/loss data...")
        np.save(self.data_path / Path("loss" + str(r) + ".npy"), self.losses)
        np.save(self.data_path / Path("reward" + str(r) + ".npy"), self.rewards)
        np.save(self.data_path / Path("predictions" + str(r) + ".npy"), self.predictions)
        print("Saved!")

        print("Testing Agent on Predefined States:")
        test_states(self.agent)
        print("Done.")

        tck = interpolate.splrep(rounds, self.losses, k=5, s=1e8)
        smoothed_loss = interpolate.splev(rounds, tck, der=0)

        tck = interpolate.splrep(rounds, self.rewards, k=5, s=1e6)
        smoothed_reward = interpolate.splev(rounds, tck, der=0)

        ncols = 3
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(1 + 9 * ncols, 9 * nrows))

        # rolling mean over n samples
        nrm = 10

        rm_rounds = rounds[nrm // 2 - 1: -nrm // 2]

        axes[0][0].plot(rounds, self.losses, alpha=0.3)
        axes[0][0].plot(rm_rounds, rolling_mean(self.losses, nrm))
        axes[0][0].set_title(f"Loss and Rolling Mean over {nrm} samples")
        axes[0][0].set_xlabel("Episode")
        axes[0][0].set_ylabel("Loss")

        axes[0][1].plot(rounds, self.rewards, alpha=0.3)
        axes[0][1].plot(rm_rounds, rolling_mean(self.rewards, nrm))
        axes[0][1].set_title(f"Reward and Rolling Mean over {nrm} samples")
        axes[0][1].set_xlabel("Episode")
        axes[0][1].set_ylabel("Reward")

        p = np.array(self.predictions)
        lp = np.array(self.predictions)

        for i, action in enumerate(self.agent.actions):
            axes[0][2].scatter(np.arange(p.shape[0]), p[:, i], label=action, marker=".")
            axes[1][2].plot(np.arange(lp.shape[0]), lp[:, i], label=action)
        axes[0][2].set_title("Action Predictions")
        axes[0][2].set_xlabel("Action")
        axes[0][2].set_ylabel("Probability")
        axes[0][2].legend()

        axes[1][0].plot(rounds, self.epsilons)
        axes[1][0].set_title("Epsilon")
        axes[1][0].set_xlabel("Episode")
        axes[1][0].set_ylabel("Epsilon")

        axes[1][1].scatter(rounds, self.steps_taken, color="C0", marker=".", alpha=0.3)
        axes[1][1].plot(rm_rounds, rolling_mean(self.steps_taken, nrm), color="C1")
        axes[1][1].set_title(f"Steps Taken Per Episode and Rolling Mean over {nrm} samples")
        axes[1][1].set_xlabel("Episode")
        axes[1][1].set_ylabel("Steps")

        axes[1][2].set_title("Action Predicitons in Last Episode")
        axes[1][2].set_xlabel("Action")
        axes[1][2].set_ylabel("Probability")

        plt.savefig(self.figures_path / Path(str(r) + ".png"))
        plt.close()

        print("Done!")

    self.predictions.clear()
