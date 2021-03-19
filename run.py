
import subprocess
import sys


def main():

    mode = "play"
    train = 1
    agents = [
        "lars_agent",
        # "rule_based_agent",
        # "rule_based_agent",
        # "rule_based_agent"
    ]

    gui = False

    n_rounds = 100000

    args = [sys.executable, "main.py", mode, "--agents", *agents, "--train", str(train), "--n-rounds", str(n_rounds)]

    if not gui:
        args.append("--no-gui")

    subprocess.run(args)


if __name__ == "__main__":
    main()
