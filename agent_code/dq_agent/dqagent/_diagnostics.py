
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ..utils import rolling_mean


class Diagnostics:
    def __init__(self):
        # per episode
        self.coins = []
        self.suicides = []
        self.kills = []
        self.deaths = []

    def add_episode(self):
        self.coins.append(0)
        self.suicides.append(0)
        self.kills.append(0)
        self.deaths.append(0)

    def __save(self, path):
        path.mkdir(exist_ok=True)
        np.save(path / Path("coins.npy"), np.asarray(self.coins))
        np.save(path / Path("suicides.npy"), np.asarray(self.suicides))
        np.save(path / Path("kills.npy"), np.asarray(self.kills))
        np.save(path / Path("deaths.npy"), np.asarray(self.deaths))

    def save(self, path, r=None, plot=True):
        self.__save(path / Path("diagnostics_data"))

        if not plot:
            return

        if r is None:
            raise ValueError("Expected to receive episode number when plot=True")

        # rolling mean over n samples
        nrm = 5000

        if nrm >= r:
            return

        path = path / Path("diagnostics")
        path.mkdir(exist_ok=True)

        ncols = 2
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(1 + 9 * ncols, 9 * nrows))

        rounds = 1 + np.arange(r)
        rm_rounds = rounds[nrm // 2 - 1: -nrm // 2]

        axes[0][0].plot(rm_rounds, rolling_mean(self.coins, nrm))
        axes[0][0].set_title("Smoothed Coins")
        axes[0][0].set_xlabel("Episode")

        axes[0][1].plot(rm_rounds, rolling_mean(self.suicides, nrm))
        axes[0][1].set_title("Smoothed Suicide Rate")
        axes[0][1].set_xlabel("Episode")

        axes[1][0].plot(rm_rounds, rolling_mean(self.kills, nrm))
        axes[1][0].set_title("Smoothed Kills")
        axes[1][0].set_xlabel("Episode")

        axes[1][1].plot(rm_rounds, rolling_mean(self.deaths, nrm))
        axes[1][1].set_title("Smoothed Death Rate (by others)")
        axes[1][1].set_xlabel("Episode")

        plt.tight_layout()
        plt.savefig(path / Path(str(r) + ".png"))
        plt.close()
