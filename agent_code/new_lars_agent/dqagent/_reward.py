from typing import List
import events as e


def reward(events: List) -> float:
    other = -3

    event_rewards = {
        e.INVALID_ACTION: -20,
        e.WAITED: -20,
        e.COIN_COLLECTED: 200,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -300
    }

    rewards = list(map(event_rewards.get, events))
    rewards = [r if r is not None else other for r in rewards]

    return sum(rewards)
