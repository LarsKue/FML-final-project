from typing import Optional

import numpy as np
import settings as s


def box(x, y, w, h, xlim, ylim):
    xmin = x - w
    xmax = x + w
    ymin = y - h
    ymax = y + h
    if xmin < 0:
        xmin = 0
        xmax = 2 * w
    if xmax >= xlim:
        xmin = xlim - 2 * w - 1
        xmax = xlim - 1
    if ymin < 0:
        ymin = 0
        ymax = 2 * h
    if ymax >= ylim:
        ymin = ylim - 2 * h - 1
        ymax = ylim - 1

    result = np.zeros((xlim, ylim), dtype=bool)

    for _x in range(xmin, xmax + 1):
        for _y in range(ymin, ymax + 1):
            result[_x, _y] = True

    return result


def observation_old(state: dict) -> np.ndarray:
    # even different approach:
    # 5x5x6 array of closest things
    result = np.squeeze(observation_old(state))

    p = state["self"][3]

    # only the closest entries
    w = 4
    h = 4
    b = box(p[0], p[1], w, h, s.COLS, s.ROWS)

    result = result[b]

    result = result.reshape((2 * w + 1, 2 * h + 1, 6))

    return np.expand_dims(result, axis=0)


def observation(state: Optional[dict]) -> np.ndarray:
    # different approach:
    # like with chess, transform the state into
    # an M x N x K board with each K-layer
    # being a flag for what is there:
    # walls
    # crates
    # bombs
    # coins
    # the player
    # enemies

    if state is None:
        # zero observation works with Q-Learning
        # since the labels are calculated as
        # y_true = reward + gamma * Q(s, a)
        # with zero observation, the model predicts all zeros for Q
        # ==> y_true = reward, which is what we want
        return np.zeros((1, s.COLS, s.ROWS, 6), dtype=float)

    # walls and crates are given by the 'field' in the state
    walls = (state["field"] == -1).astype(float)
    crates = (state["field"] == 1).astype(float)

    # bombs and coins are individual positions
    bombs = np.zeros((s.COLS, s.ROWS), dtype=float)
    coins = np.zeros((s.COLS, s.ROWS), dtype=float)

    for b in state["bombs"]:
        bombs[b[0][0], b[0][1]] = 4 - b[1]

    for c in state["coins"]:
        coins[c[0], c[1]] = 1

    player = np.zeros((s.COLS, s.ROWS), dtype=float)

    p = state["self"][3]

    player[p[0], p[1]] = 1.0 if state["self"][2] else -1

    enemies = np.zeros((s.COLS, s.ROWS), dtype=float)

    for o in state["others"]:
        op = o[3]
        enemies[op[0], op[1]] = 1.0

    result = np.stack([
        walls,
        crates,
        bombs,
        coins,
        player,
        enemies
    ], axis=-1)

    return np.expand_dims(result, axis=0)
