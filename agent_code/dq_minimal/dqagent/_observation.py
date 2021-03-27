from typing import Optional
import numpy as np
import settings as s


def observation(state: Optional[dict]) -> np.ndarray:
    if state is None:
        return np.zeros((1, s.COLS, s.ROWS, 6), dtype=float)

    walls = (state["field"] == -1).astype(float)
    crates = (state["field"] == 1).astype(float)

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
