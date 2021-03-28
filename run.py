
import subprocess
import sys
import json
from pathlib import Path
import numpy as np
import time


def make_args(mode, train, agents, gui, save_replay, n_rounds, replay_file):

    if mode == "replay":
        args = [sys.executable, "main.py", mode, replay_file]
    else:
        args = [sys.executable, "main.py", mode, "--agents", *agents, "--train", str(train), "--n-rounds", str(n_rounds)]

    if not gui:
        args.append("--no-gui")

    if save_replay:
        args.append("--save-replay")

    return args


def run_many():

    mode = "play"
    train = 1
    agents = [
        "new_lars_agent",
        # "rule_based_agent",
        # "rule_based_agent",
        # "rule_based_agent"
    ]
    gui = False
    save_replay = False
    n_rounds = int(5e5)
    replay_file = None

    settings_path = Path("agent_code/new_lars_agent/settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)

    use_existing_model = False
    overwrite = True

    epsilon = 1.0
    learning_rates = np.linspace(1e-3, 1e-4, 4)
    gammas = np.linspace(0.5, 0.9, 5)

    ##########################
    settings["setup"]["use_existing_model"] = use_existing_model
    settings["setup"]["overwrite"] = overwrite
    settings["train"]["epsilon"] = epsilon

    processes = []

    for i, lr in enumerate(learning_rates):
        for j, g in enumerate(gammas):
            n = j + len(gammas) * i
            settings["setup"]["path"] = f"test_{n}"

            settings["train"]["gamma"] = g
            settings["train"]["learning_rate"] = lr

            with open(settings_path, "w") as f:
                json.dump(settings, f)

            args = make_args(mode, train, agents, gui, save_replay, n_rounds, replay_file)

            # open the process without waiting
            p = subprocess.Popen(args, shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)

            processes.append(p)

            # sleep a couple seconds to avoid race conditions in setup
            time.sleep(5)

    for p in processes:
        p.wait()


def run_single():
    use_existing_model = False
    overwrite = True

    epsilon = 1.0
    learning_rate = 0.65
    gamma = 0.85

    mode = "play"
    train = 1
    agents = [
        "new_lars_agent",
        # "new_lars_agent",
        # "new_lars_agent",
        # "new_lars_agent",
        "rule_based_agent",
        "rule_based_agent",
        "rule_based_agent"
    ]

    gui = False
    save_replay = False
    n_rounds = int(1.5e6)
    replay_file = None

    base_path = "full_super_high_lr"

    #################################################
    settings_path = Path("agent_code/new_lars_agent/settings.json")

    if settings_path.is_file():
        with open(settings_path, "r") as f:
            settings = json.load(f)

    settings["setup"]["use_existing_model"] = use_existing_model
    settings["setup"]["overwrite"] = overwrite
    settings["setup"]["path"] = base_path
    settings["train"]["epsilon"] = epsilon
    settings["train"]["gamma"] = gamma
    settings["train"]["learning_rate"] = learning_rate

    with open(settings_path, "w+") as f:
        json.dump(settings, f)

    args = make_args(mode, train, agents, gui, save_replay, n_rounds, replay_file)

    subprocess.run(args)


def main():
    run_single()
    # run_many()


if __name__ == "__main__":
    main()
