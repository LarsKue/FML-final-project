
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import json


base_path = Path("agent_code/new_lars_agent")


def rolling_mean(x, n):
    return np.convolve(x, np.ones(n), "valid") / n


def alpha_gamma():
    base_name = "test_"

    agents = [2, 3, 6, 10, 14, 15, 16, 18]

    # agents = np.arange(20)

    ncols = 2
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(1 + 9 * ncols, 9 * nrows))

    # rolling mean over n samples
    nrm = 500

    for i in agents:
        agent_path = base_path / Path(base_name + str(i))

        with open(agent_path / Path("settings.json"), "r") as f:
            settings = json.load(f)

        gamma = settings["train"]["gamma"]
        learning_rate = settings["train"]["learning_rate"]

        data_path = agent_path / Path("diagnostics_data")

        coins = np.load(data_path / Path("coins.npy"))
        suicides = np.load(data_path / Path("suicides.npy"))

        r = len(coins)
        rounds = 1 + np.arange(r)
        rm_rounds = rounds[nrm // 2 - 1: -nrm // 2]

        label = rf"$\alpha = {learning_rate:.1e}, \gamma = {gamma:.1f}$"

        axes[0].plot(rm_rounds, rolling_mean(coins, nrm), label=label)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Smoothed Coins")
        axes[0].set_title(f"Rolling Mean of Collected Coins per Episode $(n = {nrm})$")
        axes[0].legend()

        axes[1].plot(rm_rounds, rolling_mean(suicides, nrm), label=label)
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Smoothed Suicide Rate")
        axes[1].set_title(f"Rolling Mean of the Suicide Rate $(n = {nrm})$")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig("alpha_gamma_analysis.png")
    plt.show()


def full():

    agent_path = base_path / Path("full_self_play_3")

    nrm = 5000

    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(1 + 7 * ncols, 7 * nrows))

    data_path = agent_path / Path("diagnostics_data")

    coins = np.load(data_path / Path("coins.npy"))
    suicides = np.load(data_path / Path("suicides.npy"))
    kills = np.load(data_path / Path("kills.npy"))
    deaths = np.load(data_path / Path("deaths.npy"))

    r = len(coins)
    rounds = 1 + np.arange(r)
    rm_rounds = rounds[nrm // 2 - 1: -nrm // 2]

    axes[0][0].plot(rm_rounds, rolling_mean(coins, nrm))
    axes[0][0].set_xlabel("Episode")
    axes[0][0].set_ylabel("Smoothed Coins")
    axes[0][0].set_title(f"Rolling Mean of Collected Coins per Episode $(n = {nrm})$")

    axes[0][1].plot(rm_rounds, rolling_mean(suicides, nrm))
    axes[0][1].set_xlabel("Episode")
    axes[0][1].set_ylabel("Smoothed Suicide Rate")
    axes[0][1].set_title(f"Rolling Mean of the Suicide Rate $(n = {nrm})$")

    axes[1][0].plot(rm_rounds, rolling_mean(kills, nrm))
    axes[1][0].set_xlabel("Episode")
    axes[1][0].set_ylabel("Smoothed Kills")
    axes[1][0].set_title(f"Rolling Mean of Kills $(n = {nrm})$")

    axes[1][1].plot(rm_rounds, rolling_mean(deaths, nrm))
    axes[1][1].set_xlabel("Episode")
    axes[1][1].set_ylabel("Smoothed Death Rate")
    axes[1][1].set_title(f"Rolling Mean of the Death Rate by Others $(n = {nrm})$")

    plt.tight_layout()
    plt.savefig("alpha_gamma_analysis.png")
    plt.show()


def main():
    # alpha_gamma()
    full()


if __name__ == "__main__":
    main()
