from typing import List
import events as e


def reward(self, events: List) -> float:
    other = -3

    # coins task
    # event_rewards = {
    #     e.INVALID_ACTION: -20,
    #     e.WAITED: -20,
    #     e.COIN_COLLECTED: 200,
    #     e.KILLED_SELF: 0,
    #     e.GOT_KILLED: -300
    # }

    event_rewards = {
        e.INVALID_ACTION: -20,
        e.WAITED: -15,
        e.CRATE_DESTROYED: 15,
        e.COIN_FOUND: 30,
        e.COIN_COLLECTED: 200,
        e.KILLED_OPPONENT: 1000,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -500,
        e.SURVIVED_ROUND: 30
    }

    rewards = list(map(event_rewards.get, events))
    rewards = [r if r is not None else other for r in rewards]

    if e.KILLED_SELF in events:
        self.diagnostics.suicides[-1] += 1
    elif e.GOT_KILLED in events:
        self.diagnostics.deaths[-1] += 1

    for event in events:
        if event == e.COIN_COLLECTED:
            self.diagnostics.coins[-1] += 1
        if event == e.KILLED_OPPONENT:
            self.diagnostics.kills[-1] += 1

    return sum(rewards)
