
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..utils import rolling_mean


class Diagnostics:
    def __init__(self, actions):
        self.actions = actions
        self.returns = []
        self.predictions = []
        self.last_episode_predictions = []
        self.epsilons = []

    def save(self, path, r):

        path = path / Path("diagnostics")
        path.mkdir(exist_ok=True)

        ncols = 2
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(1 + 9 * ncols, 9 * nrows))

        rounds = 1 + np.arange(r)
        # rolling mean over n samples
        nrm = 30
        rm_rounds = rounds[nrm // 2 - 1: -nrm // 2]

        axes[0][0].scatter(rounds, self.returns, color="C0", marker=".")
        axes[0][0].set_title("Returns")
        axes[0][0].plot(rm_rounds, rolling_mean(self.returns, nrm), color="C1")

        p = np.asarray(self.predictions)
        lp = np.asarray(self.last_episode_predictions)

        for i, action in enumerate(self.actions):
            axes[0][1].scatter(np.arange(p.shape[0]), p[:, i], label=action)
            axes[1][1].scatter(np.arange(lp.shape[0]), lp[:, i], label=action)

        axes[0][1].set_title("Action Predictions")
        axes[0][1].legend()

        axes[1][0].plot(rounds, self.epsilons)
        axes[1][0].set_title("Epsilon")

        axes[1][1].set_title("Predictions in Last Episode")
        axes[1][1].legend()

        plt.savefig(path / Path(str(r) + ".png"))
        plt.close()
