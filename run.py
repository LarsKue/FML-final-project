
import subprocess

def main():

    mode = "play"
    train = 0
    agents = [
        "lars_agent",
        "rule_based_agent",
        "rule_based_agent",
        "rule_based_agent"
    ]

    gui = True

    args = ["python", "main.py", mode, "--agents", *agents, "--train", str(train)]

    if not gui:
        args.append("--no-gui")

    subprocess.run(args)



if __name__ == "__main__":
    main()
