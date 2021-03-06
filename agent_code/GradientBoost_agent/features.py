import numpy as np
from collections import deque

import settings



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']



def get_features(self, game_state):
    board = game_state['field']
    current_position = game_state['self'][3]

    surrounding_feature = np.ones(41) * -1
    surrounding_index = 0
    dx_array = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    for i, y in enumerate(range(current_position[1]-4, current_position[1]+4+1)):
        dx_delta = dx_array[i]

        if y < 0 or y >= settings.ROWS:
            surrounding_index += 2*dx_delta + 1
            continue

        for x in range(current_position[0]-dx_delta, current_position[0]+dx_delta + 1):
            if x < 0 or x >= settings.COLS:
                surrounding_index += 1
                continue

            surrounding_feature[surrounding_index] = board[x, y]
            surrounding_index += 1



    self_feature = np.ones(2+1)
    self_feature[0] = current_position[0]
    self_feature[1] = current_position[1]
    self_feature[2] = int(game_state['self'][2])
            

    bomb_feature = np.ones(3*4) * 100       # 100 for non-existing bombs (arbitrary value > 15 or < -15)
    for i, (bomb, timer) in enumerate(game_state['bombs']):
        bomb_feature[3*i+0] = current_position[0] - bomb[0]
        bomb_feature[3*i+1] = current_position[1] - bomb[1]
        bomb_feature[3*i+2] = timer

    
    enemy_feature = np.ones(2*3) * 101       # 101 for dead enemies (arbitrary value > 15 or < -15)
    for i, (_, _, _, enemy) in enumerate(game_state['others']):
        enemy_feature[2*i+0] = current_position[0] - enemy[0]
        enemy_feature[2*i+1] = current_position[1] - enemy[1]

    
    coin_feature = np.ones(9*2) * 102       # 102 for non-existing coins (arbitrary value > 15 or < -15)
    for i, (coin) in enumerate(game_state['coins']):
        coin_feature[2*i+0] = current_position[0] - coin[0]
        coin_feature[2*i+1] = current_position[1] - coin[1]


    return np.concatenate([
        surrounding_feature,
        self_feature,
        bomb_feature,
        enemy_feature,
        coin_feature
    ])

    






def get_features_old(self, game_state):

    board = game_state['field'].copy()
    current_position = game_state['self'][3]
    coins = game_state['coins']
    bombs = [bomb[0] for bomb in game_state['bombs']]

    difference_vectors = np.array([np.array([0, -1]), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])])

    
    features = np.zeros(14)

    for i, v in enumerate(difference_vectors):
        next_pos = current_position + v
        features[i] = board[next_pos[0], next_pos[1]] 


    paths_nearest_coins = shortest_path(board, current_position, coins)

    
    if paths_nearest_coins is not None:
        path = paths_nearest_coins[np.random.choice(np.arange(len(paths_nearest_coins)))]

        nearest_coin_position = path[-1]

        features[4] = len(path)
        
        #features[4] = nearest_coin_position[0] - current_position[0]
        #features[5] = nearest_coin_position[1] - current_position[1]

        features[5] = path[1][0] - current_position[0]
        features[6] = path[1][1] - current_position[1]




    paths_nearest_bombs = shortest_path(board, current_position, bombs)

    if paths_nearest_bombs is not None:
        path = paths_nearest_bombs[0]

        nearest_bomb_position = path[-1]

        direction = np.subtract(nearest_bomb_position, current_position)
        
        features[7] = bombs[bombs.index(nearest_bomb_position)][1]

        if (direction[0] == 0 or direction[1] == 0) and np.linalg.norm(direction) <= 4:
            features[8] = 1
        
        features[9] = direction[0]
        features[10] = direction[1]


        
    paths_nearest_crates = shortest_path_target_value(board, current_position, board, 1)

    if paths_nearest_crates is not None:
        path = paths_nearest_crates[0]

        nearest_crate_position = path[-2]

        features[11] = len(path)
        
        #features[4] = nearest_coin_position[0] - current_position[0]
        #features[5] = nearest_coin_position[1] - current_position[1]

        features[12] = path[1][0] - current_position[0]
        features[13] = path[1][1] - current_position[1]


    return features






def shortest_path(board, start, targets):
    nearest_targets = []
    
    parents = {}
    parents[start] = start

    q = deque()
    q.append((start, 0))

    append_neighbors = True
    path_length = 100000

    while len(q) > 0:
        node, length = q.popleft()
        
        if length > path_length:
            continue

        if node in targets:
            nearest_targets.append(node)
            append_neighbors = False
            path_length = length
            
        
        x, y = node
        neighbors = [(nx, ny) for nx, ny in [(x, y-1), (x+1, y), (x, y+1), (x-1, y)] if board[(nx, ny)] == 0]
        for neighbor in neighbors:
            if not neighbor in parents:
                parents[neighbor] = node
                if append_neighbors:
                    q.append((neighbor, length + 1))

    if len(nearest_targets) == 0:
        return None

    else:
        paths = []
        for t in nearest_targets:
            path = [t]
            while parents[t] != start:
                path.append(parents[t])
                t = parents[t]
            path.append(start)
            path.reverse()
            paths.append(path)

        return paths

def shortest_path_target_value(board, start, target_board, target_value, free_tiles=[0]):
    nearest_targets = []
    
    parents = {}
    parents[start] = start

    q = deque()
    q.append(start)

    append_neighbors = True

    while len(q) > 0:
        node = q.popleft()
        if target_board[node] == target_value:
            nearest_targets.append(node)
            append_neighbors = False
        
        x, y = node
        neighbors = [(nx, ny) for nx, ny in [(x, y-1), (x+1, y), (x, y+1), (x-1, y)] if board[(nx, ny)] in free_tiles]
        for neighbor in neighbors:
            if not neighbor in parents:
                parents[neighbor] = node
                if append_neighbors:
                    q.append(neighbor)

    if len(nearest_targets) == 0:
        return None

    else:
        paths = []
        for t in nearest_targets:
            path = [t]
            while parents[t] != start:
                path.append(parents[t])
                t = parents[t]
            path.append(start)
            path.reverse()
            paths.append(path)

        return paths

def new_position(pos, action):
    if action == "BOMB" or action == "WAIT":
        return pos
    elif action == "UP":
        return (pos[0], pos[1] - 1)
    elif action == "RIGHT":
        return (pos[0] + 1, pos[1])
    elif action == "DOWN":
        return (pos[0], pos[1] + 1)
    elif action == "LEFT":
        return (pos[0] - 1, pos[1])