
from typing import List
import tensorflow as tf

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
    self.learning_rate = 1e-3
    self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    self.rewards = []
    self.losses = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.agent.reward(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    loss, reward = self.agent.train(self.optimizer)

    self.rewards.append(reward)
    self.losses.append(loss)

    # save the agent to disk
    self.agent.save(self.model_path)

    r = last_game_state["round"]

    print("Finished round", r)
    if r % 100 == 0:
        import numpy as np
        import matplotlib.pyplot as plt
        print("Plotting Rewards and Losses")
        rounds = np.arange(0, r)
        plt.figure(figsize=(10, 9))
        plt.plot(rounds, self.rewards, label="Rewards")
        plt.plot(rounds, self.losses, label="Losses")
        plt.legend()
        plt.savefig(f"{r}.png")
        plt.close()
