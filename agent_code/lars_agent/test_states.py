
import numpy as np
import settings as s


def test_states(agent):

    base = {
        "round": 1,
        "step": 15,
        "field": np.zeros((s.COLS, s.ROWS)),
        "self": ("lars_agent", 0, False, (1, 1)),
        "others": [],
        "bombs": [],
        "coins": [],
        "user_input": "WAIT",
        "explosion_map": np.zeros((s.COLS, s.ROWS))
    }

    coin_left = base
    coin_left["coins"] = [(0, 1)]

    o = agent.observation(coin_left)
    _, prediction = agent.predict(o)
    print("Prediction Coin Left:", prediction)

    coin_right = base
    coin_right["coins"] = [(2, 1)]

    o = agent.observation(coin_right)
    _, prediction = agent.predict(o)
    print("Prediction Coin Right:", prediction)

    coin_top = base
    coin_top["coins"] = [(1, 0)]

    o = agent.observation(coin_top)
    _, prediction = agent.predict(o)
    print("Prediction Coin Top:", prediction)

    coin_bottom = base
    coin_bottom["coins"] = [(1, 2)]

    o = agent.observation(coin_bottom)
    _, prediction = agent.predict(o)
    print("Prediction Coin Bottom:", prediction)
