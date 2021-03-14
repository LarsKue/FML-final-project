import numpy as np
from collections import deque

import settings



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def feature_matrix(self, game_state: dict):
    return np.stack([
        feature_1(self, game_state, ACTIONS),       # + move towards coin
        feature_2(self, game_state, ACTIONS),       # + collect coin
        feature_3(self, game_state, ACTIONS),       # - invalid action
        feature_4(self, game_state, ACTIONS),       # + move out of explosion area
        feature_5(self, game_state, ACTIONS),       # - move/stay in explosion area
        feature_6(self, game_state, ACTIONS),       # + move towards crate
        feature_7(self, game_state, ACTIONS),       # - bomb is suicidal
        feature_8(self, game_state, ACTIONS),       # + bomb will destroy crates, iff not suicidal
        feature_9(self, game_state, ACTIONS),       # - go into dead end if previous action was 'BOMB'
        feature_10(self, game_state, ACTIONS),      # + move towards nearest enemy
        feature_11(self, game_state, ACTIONS),      # + place bomb if standing next to enemy and not suicidal
        feature_12(self, game_state, ACTIONS),      # + place bomb if enemy is in explosion area and not suicidal
    ], axis=0)


def feature_vector(self, game_state: dict, action):
    action_arr = [action]
    return np.array([
        feature_1(self, game_state, action_arr),
        feature_2(self, game_state, action_arr),
        feature_3(self, game_state, action_arr),
        feature_4(self, game_state, action_arr),
        feature_5(self, game_state, action_arr),
        feature_6(self, game_state, action_arr),
        feature_7(self, game_state, action_arr),
        feature_8(self, game_state, action_arr),
        feature_9(self, game_state, action_arr),
        feature_10(self, game_state, action_arr),
        feature_11(self, game_state, action_arr),
        feature_12(self, game_state, action_arr),
    ]).flatten()



def feature_1(self, game_state: dict, actions):
    # move towards coin

    board = game_state['field'].copy()
    coins = game_state['coins']
    current_position = game_state['self'][3]

    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    shortest_paths_coins = shortest_path(board, current_position, coins)
    next_positions = [new_position(current_position, a) for a in actions]

    feature1 = np.zeros(len(actions))
    if shortest_paths_coins is not None:
        best_next_positions = [path[1] for path in shortest_paths_coins]
        #path_length = len(shortest_paths_coins[0])
        
        for i, next_position in enumerate(next_positions):
            if next_position in best_next_positions:

                #if path_length < 15:
                #    feature1[i] = 3
                #elif path_length < 25:
                #    feature1[i] = 2
                #else:
                #    feature1[i] = 1
                
                feature1[i] = 1

    return feature1


def feature_2(self, game_state: dict, actions):
    # collect coin with next move
    
    coins = game_state['coins']
    current_position = game_state['self'][3]
    next_positions = [new_position(current_position, a) for a in actions]

    feature2 = np.zeros(len(actions))
    for i, next_pos in enumerate(next_positions):
        if next_pos in coins:
            feature2[i] = 1
    
    return feature2
    

def feature_3(self, game_state: dict, actions):
    # invalid action

    board = game_state['field'].copy()
    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    current_position = game_state['self'][3]

    feature3 = np.zeros(len(actions))
    for i, a in enumerate(actions):
        if a == 'WAIT':
            continue
        elif a == 'BOMB':
            if not game_state['self'][2]:
                feature3[i] = 1
            continue
        else:
            next_pos = new_position(current_position, a)
            if board[next_pos] != 0:
                feature3[i] = 1
                continue

    return feature3
        

def feature_4(self, game_state: dict, actions):
    # move out of explosion area
    # pretend every bomb will explode after action

    board = game_state['field']
    bombs = game_state['bombs']
    current_position = game_state['self'][3]

    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            if board[bomb_x + i, bomb_y] != 1:
                board[bomb_x + i, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            if board[bomb_x - i, bomb_y] != 1:
                board[bomb_x - i, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            if board[bomb_x, bomb_y + i] != 1:
                board[bomb_x, bomb_y + i] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            if board[bomb_x, bomb_y - i] != 1:
                board[bomb_x, bomb_y - i] = 5

    feature4 = np.zeros(len(actions))

    if board[current_position] != 5:
        return feature4
    
    
    paths_outside_explosion = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])
    #print("f4:", paths_outside_explosion)

    next_positions = [new_position(current_position, a) for a in actions]

    if paths_outside_explosion is not None:
        best_next_positions = [path[1] for path in paths_outside_explosion]
        for i, next_position in enumerate(next_positions):
            if next_position in best_next_positions:
                feature4[i] = 1


    return feature4


def feature_5(self, game_state: dict, actions):
    # stay in explosion area or move towards bomb
    # pretend every bomb will explode after action

    board = game_state['field']
    bombs = game_state['bombs']
    current_position = game_state['self'][3]

    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            if board[bomb_x + i, bomb_y] != 1:
                board[bomb_x + i, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            if board[bomb_x - i, bomb_y] != 1:
                board[bomb_x - i, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            if board[bomb_x, bomb_y + i] != 1:
                board[bomb_x, bomb_y + i] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            if board[bomb_x, bomb_y - i] != 1:
                board[bomb_x, bomb_y - i] = 5


    feature5 = np.zeros(len(actions))

    if board[current_position] == 0:
        for i, a in enumerate(actions):
            next_pos = new_position(current_position, a)
            if board[next_pos] == 5:
                feature5[i] = 1

        return feature5
    
    
    paths_outside_explosion = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])

    next_positions = [new_position(current_position, a) for a in actions]

    if paths_outside_explosion is not None:
        best_next_positions = [path[1] for path in paths_outside_explosion]
        for i, next_position in enumerate(next_positions):
            if next_position not in best_next_positions:
                feature5[i] = 1


    return feature5


def feature_6(self, game_state: dict, actions):
    # move towards crate
    
    board = game_state['field'].copy()
    current_position = game_state['self'][3]

    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    shortest_paths_crates = shortest_path_target_value(board, current_position, board, 1, free_tiles=[0,1])
    next_positions = [new_position(current_position, a) for a in actions]

    feature6 = np.zeros(len(actions))
    if shortest_paths_crates is not None:
        best_next_positions = [path[1] for path in shortest_paths_crates]
        for i, next_position in enumerate(next_positions):
            if next_position in best_next_positions:
                feature6[i] = 1

    return feature6



def feature_7(self, game_state: dict, actions):
    # bomb is suicidal

    if 'BOMB' not in actions:
        return np.zeros(len(actions))

    
    feature7 = np.zeros(len(actions))

    
    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()
    current_position = game_state['self'][3]

    (bomb_x, bomb_y) = current_position
    board[bomb_x, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x + i, bomb_y] == -1:
            break
        if board[bomb_x + i, bomb_y] != 1:
            board[bomb_x + i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x - i, bomb_y] == -1:
            break
        if board[bomb_x - i, bomb_y] != 1:
            board[bomb_x - i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y + i] == -1:
            break
        if board[bomb_x, bomb_y + i] != 1:
            board[bomb_x, bomb_y + i] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y - i] == -1:
            break
        if board[bomb_x, bomb_y - i] != 1:
            board[bomb_x, bomb_y - i] = 5


    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4



    shortest_path_to_safety = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])

    if shortest_path_to_safety is not None:
        # path to safe spot exits
        return feature7
    

    BOMB_index = 0 if len(actions) == 1 else 5
    feature7[BOMB_index] = 1

    return feature7


def feature_8(self, game_state: dict, actions):
    # bomb will destroy crates (iff not suicidal)

    if 'BOMB' not in actions:
        # for feature_vector
        return np.zeros(len(actions))

    feature8 = np.zeros(len(actions))

    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()
    current_position = game_state['self'][3]

    
    hit_crate = False
    (bomb_x, bomb_y) = current_position
    board[bomb_x, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x + i, bomb_y] == -1:
            break
        if board[bomb_x + i, bomb_y] == 1:
            hit_crate = True
        else:
            board[bomb_x + i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x - i, bomb_y] == -1:
            break
        if board[bomb_x - i, bomb_y] == 1:
            hit_crate = True
        else:
            board[bomb_x - i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y + i] == -1:
            break
        if board[bomb_x, bomb_y + i] == 1:
            hit_crate = True
        else:
            board[bomb_x, bomb_y + i] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y - i] == -1:
            break
        if board[bomb_x, bomb_y - i] == 1:
            hit_crate = True
        else:
            board[bomb_x, bomb_y - i] = 5

    if not hit_crate:
        return feature8


    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4


    shortest_path_to_safety = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])

    if shortest_path_to_safety is None:
        # path to safe spot doesn't exits
        return feature8



    BOMB_index = 0 if len(actions) == 1 else 5
    feature8[BOMB_index] = 1

    return feature8


def feature_9(self, game_state: dict, actions):
    # walk into dead_end (only if previous action was 'BOMB')

    feature9 = np.zeros(len(actions))

    if self.previous_action != 'BOMB':
        return feature9

        
    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()
    current_position = game_state['self'][3]
    

    (bomb_x, bomb_y) = current_position
    board[bomb_x, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x + i, bomb_y] == -1:
            break
        if board[bomb_x + i, bomb_y] != 1:
            board[bomb_x + i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x - i, bomb_y] == -1:
            break
        if board[bomb_x - i, bomb_y] != 1:
            board[bomb_x - i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y + i] == -1:
            break
        if board[bomb_x, bomb_y + i] != 1:
            board[bomb_x, bomb_y + i] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y - i] == -1:
            break
        if board[bomb_x, bomb_y - i] != 1:
            board[bomb_x, bomb_y - i] = 5


    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    
    for (bomb_x, bomb_y), _ in bombs:
        if (bomb_x, bomb_y) == current_position:
            continue
        board[bomb_x, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4

    for i, a in enumerate(actions):
        if a == 'BOMB' or a == 'WAIT':
            continue
            
        
        next_position = new_position(current_position, a)
        if board[next_position] != 5:
            continue
        
        shortest_path_to_safety = shortest_path_target_value(board, next_position, board, 0, free_tiles=[0,5])
        
        #print("action:", a, "path:", shortest_path_to_safety)

        if shortest_path_to_safety is None:
            feature9[i] = 1

    #print("")
    #print(feature9)

    return feature9


def feature_10(self, game_state: dict, actions):
    # move towards nearest enemy

    board = game_state['field'].copy()
    current_position = game_state['self'][3]

    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    enemies = [enemy[3] for enemy in game_state['others']]

    shortest_paths_enemy = shortest_path(board, current_position, enemies)
    next_positions = [new_position(current_position, a) for a in actions]

    feature10 = np.zeros(len(actions))
    if shortest_paths_enemy is not None:
        best_next_positions = [path[1] for path in shortest_paths_enemy]
        for i, next_position in enumerate(next_positions):
            if next_position in best_next_positions:
                feature10[i] = 1

    return feature10



def feature_11(self, game_state: dict, actions):
    # place bomb if standing next to enemy, if not suicidal

    if 'BOMB' not in actions:
        return np.zeros(len(actions))


    current_position = game_state['self'][3]
    

    feature11 = np.zeros(len(actions))

    enemies = [enemy[3] for enemy in game_state['others']]
    standing_next_to_enemy = False
    for a in ['UP', 'RIGHT', 'DOWN', 'LEFT']:
        if new_position(current_position, a) in enemies:
            standing_next_to_enemy = True
            break
    
    if not standing_next_to_enemy:
        return feature11

    
    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()

    (bomb_x, bomb_y) = current_position
    board[bomb_x, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x + i, bomb_y] == -1:
            break
        if board[bomb_x + i, bomb_y] != 1:
            board[bomb_x + i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x - i, bomb_y] == -1:
            break
        if board[bomb_x - i, bomb_y] != 1:
            board[bomb_x - i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y + i] == -1:
            break
        if board[bomb_x, bomb_y + i] != 1:
            board[bomb_x, bomb_y + i] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y - i] == -1:
            break
        if board[bomb_x, bomb_y - i] != 1:
            board[bomb_x, bomb_y - i] = 5


    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4



    shortest_path_to_safety = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])

    if shortest_path_to_safety is None:
        # path to safe spot doesn't exits
        return feature11
    

    BOMB_index = 0 if len(actions) == 1 else 5
    feature11[BOMB_index] = 1

    return feature11


def feature_12(self, game_state: dict, actions):
    # place bomb that could kill enemy (inside explosion area), if not suicidal

    if 'BOMB' not in actions:
        return np.zeros(len(actions))

    
    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()
    current_position = game_state['self'][3]

    

    feature12 = np.zeros(len(actions))

    enemies = [enemy[3] for enemy in game_state['others']]
    enemy_inside_explosion_area = False
    
    (bomb_x, bomb_y) = current_position
    board[bomb_x, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x + i, bomb_y] == -1:
            break
        if (bomb_x + i, bomb_y) in enemies:
            enemy_inside_explosion_area = True
        if board[bomb_x + i, bomb_y] != 1:
            board[bomb_x + i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x - i, bomb_y] == -1:
            break
        if (bomb_x - i, bomb_y) in enemies:
            enemy_inside_explosion_area = True
        if board[bomb_x - i, bomb_y] != 1:
            board[bomb_x - i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y + i] == -1:
            break
        if (bomb_x, bomb_y + i) in enemies:
            enemy_inside_explosion_area = True
        if board[bomb_x, bomb_y + i] != 1:
            board[bomb_x, bomb_y + i] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y - i] == -1:
            break
        if (bomb_x, bomb_y - i) in enemies:
            enemy_inside_explosion_area = True
        if board[bomb_x, bomb_y - i] != 1:
            board[bomb_x, bomb_y - i] = 5

        
    if not enemy_inside_explosion_area:
        return feature12



    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4



    shortest_path_to_safety = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])

    if shortest_path_to_safety is None:
        # path to safe spot doesn't exits
        return feature12
    

    BOMB_index = 0 if len(actions) == 1 else 5
    feature12[BOMB_index] = 1

    return feature12












def feature_10_old(self, game_state: dict, actions):
    # all actions will lead to sd
    
    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()
    current_position = game_state['self'][3]


    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4



    shortest_path_to_safety = shortest_path_target_value(board, current_position, board, 0)

    if shortest_path_to_safety is not None:
        # path to safe spot exits
        return np.zeros(len(actions))
    else:
        return np.ones(len(actions))
    
def feature_7_old(self, game_state: dict, actions):
    # place bomb if agent can escape
    
    if 'BOMB' not in actions:
        # for feature_vector
        return np.zeros(len(actions))

    feature7 = np.zeros(len(actions))


    board = game_state['field'].copy()
    bombs = game_state['bombs'].copy()
    current_position = game_state['self'][3]

    hit_crate = False
    (bomb_x, bomb_y) = current_position
    board[bomb_x, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x + i, bomb_y] == -1:
            break
        if board[bomb_x + i, bomb_y] == 1:
            hit_crate = True
        else:
            board[bomb_x + i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x - i, bomb_y] == -1:
            break
        if board[bomb_x - i, bomb_y] == 1:
            hit_crate = True
        else:
            board[bomb_x - i, bomb_y] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y + i] == -1:
            break
        if board[bomb_x, bomb_y + i] == 1:
            hit_crate = True
        else:
            board[bomb_x, bomb_y + i] = 5
    for i in range(1, settings.BOMB_POWER + 1):
        if board[bomb_x, bomb_y - i] == -1:
            break
        if board[bomb_x, bomb_y - i] == 1:
            hit_crate = True
        else:
            board[bomb_x, bomb_y - i] = 5

    if not hit_crate:
        return feature7


    for enemy in game_state['others']:
        board[enemy[3][0], enemy[3][1]] = 2
    for bomb in game_state['bombs']:
        board[bomb[0][0], bomb[0][1]] = 3

    
    for (bomb_x, bomb_y), _ in bombs:
        board[bomb_x, bomb_y] = 5
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x + i, bomb_y] == -1:
                break
            board[bomb_x + i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x - i, bomb_y] == -1:
                break
            board[bomb_x - i, bomb_y] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y + i] == -1:
                break
            board[bomb_x, bomb_y + i] = 4
        for i in range(1, settings.BOMB_POWER + 1):
            if board[bomb_x, bomb_y - i] == -1:
                break
            board[bomb_x, bomb_y - i] = 4



    shortest_path_to_safety = shortest_path_target_value(board, current_position, board, 0, free_tiles=[0,5])

    #if not game_state['self'][2]:
    #    print("path to safety:", shortest_path_to_safety)

    if shortest_path_to_safety is None:
        #print("no shortest path...")
        #print(feature7)
        return feature7
    
    #print(shortest_path_to_safety)

    
    BOMB_index = 0 if len(actions) == 1 else 5
    feature7[BOMB_index] = 1

    return feature7

def feature_8_old(self, game_state: dict, actions):
    return np.zeros(len(actions))




def shortest_path(board, start, targets):
    nearest_targets = []
    
    parents = {}
    parents[start] = start

    q = deque()
    q.append(start)

    append_neighbors = True

    while len(q) > 0:
        node = q.popleft()
        if node in targets:
            nearest_targets.append(node)
            append_neighbors = False
        
        x, y = node
        neighbors = [(nx, ny) for nx, ny in [(x, y-1), (x+1, y), (x, y+1), (x-1, y)] if board[(nx, ny)] == 0]
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