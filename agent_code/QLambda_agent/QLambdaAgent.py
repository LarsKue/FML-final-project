from typing import List
import numpy as np
from collections import deque
from scipy.special import softmax

import pickle
import settings as s

import events as e
import settings


class QLambdaAgent:

    def __init__(self, train=False, weights=None):
        self.num_features = 14
        # self.weights = np.random.uniform(-1, 1, 13).T
        # self.weights = np.array([2, 2, -1, 3, -3, 1, -1, 3, -1, 1, 1, 1, 1, -1], dtype=float).T
        # self.weights = np.array([1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1], dtype=float).T
        if weights is None:
            self.weights = np.array([1, 1, -1, 1, -1, 1, -1, 1, 0, 1, 1, 1, 0, -1], dtype=float).T
            self.weights = np.zeros(self.num_features)
        else:
            self.weights = weights

        self.train = train

        self.previous_action = None
        self.current_action = None

        self.WALL_VALUE = -1
        self.FREE_VALUE = 0
        self.CRATE_VALUE = 1
        self.ENEMY_VALUE = 2
        self.BOMB_VALUE = 3
        self.ENEMY_EXPLOSION_VALUE = 4
        self.OWN_EXPLOSION_VALUE = 5
        self.COIN_VALUE = 6
        self.DEAD_END_VALUE = 7

        self.action_indices = {
            'UP': 0,
            'RIGHT': 1,
            'DOWN': 2,
            'LEFT': 3,
            'WAIT': 4,
            'BOMB': 5,
        }
        self.actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

        if self.train:
            self.epsilon = 0.15
            self.epsilon_decrease = 0.997
            self.min_epsilon = 0.05

            self.alpha = 0.001
            self.discount = 0.95
            self.lambda_et = 0.95

            self.action_was_random = False

            self.state_action_count = {}
            self.temperature = 1.0

            self.experience_buffer = [[]]

            self.update_number = 0
            self.weights_updates = np.zeros((2001, self.num_features))
            self.weights_updates[0] = self.weights

            # self.observations = []
            # self.observation_save = 0
            # self.new_observation = None
            # self.chosen_action = []
            # self.event_array = []

    def act(self, game_state: dict):
        if game_state['step'] == 1:
            self.previous_action = None
            self.current_action = None
        self.previous_action = self.current_action
        action = None
        if self.train:
            if np.random.uniform() < self.epsilon:
                self.action_was_random = True
                action = self._explore(game_state)
            else:
                self.action_was_random = False
                action = self._exploit(game_state)
        else:
            action = self._exploit(game_state)

        if self._action_is_valid(game_state, action) and not (self.previous_action == 'BOMB' and action == 'WAIT'):
            self.current_action = action
        return action

    def add_sars(self, old_game_state: dict, action: str, new_game_state: dict, events: List[str]):
        if old_game_state is None:
            self.name = new_game_state['self'][0]
            old_game_state_features = self.experience_buffer[-1][-1][3]
        else:
            old_game_state_features = self._get_features(old_game_state)

        self.experience_buffer[-1].append([
            old_game_state_features,
            action,
            self._calculate_reward(events),
            self._get_features(new_game_state),
            self.action_was_random
        ])

        """
        if old_game_state is None:
            old_game_state_observation = self.new_observation
            self.new_observation = None
        else:
            old_game_state_observation = self._observation(old_game_state)
            self.new_observation = self._observation(new_game_state)

        self.observations.append(old_game_state_observation)
        self.chosen_action.append(action)
        self.event_array.append(", ".join(events))

        if old_game_state is None and new_game_state['round'] % 500 == 0:
            self.observations = np.array(self.observations)
            self.observation_save += 1
            np.save(f"training_data/observations_{self.observation_save}", self.observations)

            with open(f"training_data/actions_{self.observation_save}", "wb") as fp:
                pickle.dump(self.chosen_action, fp)

            with open(f"training_data/events_{self.observation_save}", "wb") as fp:
                pickle.dump(self.event_array, fp)

            self.observations = []
            self.chosen_action = []
            self.event_array = []
        """

    def update_weights(self):
        self._Watkins_QLambda()

        self.epsilon *= self.epsilon_decrease
        self.epsilon = max(self.epsilon, self.min_epsilon)

        self.update_number += 1
        self.weights_updates[self.update_number] = self.weights

        self.experience_buffer = [[]]

        if self.update_number > 0 and self.update_number % 10 == 0:
            np.save(f"weights_updates_{self.name}.npy", self.weights_updates)

    def print_weights(self):
        print("")
        for w in self.weights:
            print(f"{w:0.3f}, ", sep="", end="")

        print("\n", self.epsilon, self.alpha)

    def _softmax(self, x):
        return (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))))

    def _explore(self, game_state: dict):
        features = self._get_features(game_state)
        features_tuple = tuple(features.flatten())
        if features_tuple in self.state_action_count:
            probabilities = self._softmax(-self.state_action_count[features_tuple])
        else:
            probabilities = np.ones(len(self.actions)) / len(self.actions)

        action = np.random.choice(self.actions, p=probabilities)

        if features_tuple in self.state_action_count:
            self.state_action_count[features_tuple][self.action_indices[action]] += 1
        else:
            count = np.zeros(len(self.actions))
            count[self.action_indices[action]] = 1
            self.state_action_count[features_tuple] = count
        
        return action

    def _exploit(self, game_state: dict):
        features = self._get_features(game_state)

        q = self.weights.dot(features)
        best_actions = np.argwhere(q == np.amax(q)).flatten()

        action = self.actions[np.random.choice(best_actions)]

        if self.train:
            features_tuple = tuple(features.flatten())
            if features_tuple in self.state_action_count:
                self.state_action_count[features_tuple][self.action_indices[action]] += 1
            else:
                count = np.zeros(len(self.actions))
                count[self.action_indices[action]] = 1
                self.state_action_count[features_tuple] = count

        return action

    def _calculate_reward(self, events: List[str]):
        reward = 0

        for event in events:
            if event == e.INVALID_ACTION:
                reward += -200
            elif event == e.CRATE_DESTROYED:
                reward += 10
            elif event == e.COIN_FOUND:
                reward += 30
            elif event == e.COIN_COLLECTED:
                reward += 150
            elif event == e.KILLED_OPPONENT:
                reward += 1000
            elif event == e.KILLED_SELF:
                reward += 0              # sd == KILLED_SELF + GOT_KILLED => reward(sd) = -100
            elif event == e.GOT_KILLED:
                reward += -1000

        return reward / 100

    def _Watkins_QLambda(self):
        for episode in self.experience_buffer:
            eligibility_trace = np.zeros(self.num_features)

            for old_state, action, reward, new_state, action_was_random in episode:
                eligibility_trace += old_state[:, self.action_indices[action]]

                delta = reward - self.weights.dot(old_state[:, self.action_indices[action]])
                delta += self.discount * np.max(self.weights.dot(new_state))

                self.weights += self.alpha * delta * eligibility_trace

                if action_was_random:
                    eligibility_trace = np.zeros(self.num_features)
                else:
                    eligibility_trace *= self.discount * self.lambda_et

    def _get_features(self, game_state: dict):
        coins = game_state['coins']
        enemies = [enemy[3] for enemy in game_state['others']]
        bombs = game_state['bombs']
        current_position = game_state['self'][3]
        bomb_is_possible = game_state['self'][2]
        next_positions = [self._position_after_action(current_position, action) for action in self.actions]
        board = game_state['field'].copy()

        board_with_dead_ends = self._get_dead_ends(board)

        # board:
        #   -1: wall
        #   0:  free
        #   1:  crate
        #   2:  enemy
        #   3:  bomb
        #   4:  explosion area (assume that bomb will explode next step)
        #   5:  explosion area of own bomb, placed on current_position (if possible)

        for bomb, timer in bombs:
            board[bomb] = self.BOMB_VALUE
        for enemy in enemies:
            board[enemy] = self.ENEMY_VALUE

        board_without_explosion = board.copy()

        board_with_coins = board.copy()
        for coin in coins:
            board_with_coins[coin] = self.COIN_VALUE

        explosion_timer_map = {}

        for (bomb_x, bomb_y), timer in bombs:
            explosion_timer_map[(bomb_x, bomb_y)] = min(timer, explosion_timer_map.get((bomb_x, bomb_y), 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x + i, bomb_y)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x - i, bomb_y)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x, bomb_y + i)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x, bomb_y - i)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 3))

        board_with_own_bomb = board.copy()
        board_only_own_bomb = board_without_explosion.copy()
        hit_crates = False
        enemy_inside_explosion_area = False
        if bomb_is_possible:
            # placing bomb is possible
            bomb_x, bomb_y = current_position
            board_with_own_bomb[current_position] = 3
            board_only_own_bomb[current_position] = 3
            explosion_timer_map[current_position] = min(3, explosion_timer_map.get(current_position, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x + i, bomb_y)
                if board_with_own_bomb[coord] == self.WALL_VALUE:
                    break
                if board_with_own_bomb[coord] == self.FREE_VALUE:
                    board_with_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_only_own_bomb[coord] == self.FREE_VALUE:
                    board_only_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_with_own_bomb[coord] == self.CRATE_VALUE:
                    hit_crates = True
                if board_with_own_bomb[coord] == self.ENEMY_VALUE:
                    enemy_inside_explosion_area = True
                    explosion_timer_map[coord] = min(3, explosion_timer_map.get(coord, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x - i, bomb_y)
                if board_with_own_bomb[coord] == self.WALL_VALUE:
                    break
                if board_with_own_bomb[coord] == self.FREE_VALUE:
                    board_with_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_only_own_bomb[coord] == self.FREE_VALUE:
                    board_only_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_with_own_bomb[coord] == self.CRATE_VALUE:
                    hit_crates = True
                if board_with_own_bomb[coord] == self.ENEMY_VALUE:
                    enemy_inside_explosion_area = True
                    explosion_timer_map[coord] = min(3, explosion_timer_map.get(coord, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x, bomb_y + i)
                if board_with_own_bomb[coord] == self.WALL_VALUE:
                    break
                if board_with_own_bomb[coord] == self.FREE_VALUE:
                    board_with_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_only_own_bomb[coord] == self.FREE_VALUE:
                    board_only_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_with_own_bomb[coord] == self.CRATE_VALUE:
                    hit_crates = True
                if board_with_own_bomb[coord] == self.ENEMY_VALUE:
                    enemy_inside_explosion_area = True
                    explosion_timer_map[coord] = min(3, explosion_timer_map.get(coord, 3))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x, bomb_y - i)
                if board_with_own_bomb[coord] == self.WALL_VALUE:
                    break
                if board_with_own_bomb[coord] == self.FREE_VALUE:
                    board_with_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_only_own_bomb[coord] == self.FREE_VALUE:
                    board_only_own_bomb[coord] = self.OWN_EXPLOSION_VALUE
                if board_with_own_bomb[coord] == self.CRATE_VALUE:
                    hit_crates = True
                if board_with_own_bomb[coord] == self.ENEMY_VALUE:
                    enemy_inside_explosion_area = True
                    explosion_timer_map[coord] = min(3, explosion_timer_map.get(coord, 3))

        return np.stack([
            # + move towards coin
            self._feature_1(board_with_coins, board_without_explosion, current_position, next_positions),
            # + collect coin
            self._feature_2(board_with_coins, current_position, next_positions),
            # - invalid action
            self._feature_3(board_without_explosion, current_position, bomb_is_possible),
            # + move out of explosion area
            self._feature_4(board, current_position, next_positions),
            # - move/stay in explosion area
            self._feature_5(board, explosion_timer_map, current_position, next_positions),
            # + move towards crate
            self._feature_6(board_without_explosion, current_position, next_positions),
            # - bomb is suicidal
            self._feature_7(board_with_own_bomb, current_position, bomb_is_possible),
            # + bomb will destroy crates
            self._feature_8(board, current_position, bomb_is_possible, hit_crates),
            # - go into dead end if previous action was 'BOMB'
            self._feature_9(board_only_own_bomb, current_position, next_positions),
            # + move towards nearest enemy
            self._feature_10(board_without_explosion, current_position, next_positions),
            # + place bomb if standing next to enemy
            self._feature_11(board, current_position, bomb_is_possible, next_positions),
            # + place bomb if enemy is in explosion area
            self._feature_12(board_only_own_bomb, current_position, bomb_is_possible, enemy_inside_explosion_area),
            # + place bomb if enemy is in explosion area and in dead end
            self._feature_13(board_with_dead_ends, enemies, current_position, bomb_is_possible, enemy_inside_explosion_area),
            # - go into dead end if enemy is nearby
            self._feature_14(board_with_dead_ends, board_without_explosion, current_position, next_positions)
        ], axis=0)

    def _feature_1_old(self, board_with_coins, current_position, next_positions):
        # move towards nearest coin
        shortest_path_coins = self._shortest_path(board_with_coins, current_position, self.COIN_VALUE, free_tiles=(self.FREE_VALUE, self.COIN_VALUE))

        feature1 = np.zeros(len(self.actions))

        if shortest_path_coins is not None:
            best_next_positions = [path[1] for path in shortest_path_coins]
            path_length = len(shortest_path_coins[0])

            for i, next_position in enumerate(next_positions):
                if next_position in best_next_positions:
                    if path_length <= 10:
                        feature1[i] = 1
                    elif path_length <= 15:
                        feature1[i] = 0.75
                    else:
                        feature1[i] = 0.5
                    # feature1[i] = 1

        return feature1

    def _feature_1(self, board_with_coins, board_without_explosion, current_position, next_positions):
        # number of steps to coin in direction

        feature1 = np.zeros(len(self.actions))

        value_of_current_position = board_with_coins[current_position]
        board_with_coins[current_position] = 10     # cannot go back to this tile

        for i, next_position in enumerate(next_positions):
            if self.actions[i] == 'WAIT' or self.actions[i] == 'BOMB':
                continue
            if board_without_explosion[next_position] != self.FREE_VALUE:
                continue
            shortest_path_coins = self._shortest_path(board_with_coins, next_position, self.COIN_VALUE, free_tiles=(self.FREE_VALUE, self.COIN_VALUE))

            if shortest_path_coins is None:
                continue

            path_length = len(shortest_path_coins[0]) + 1
            feature1[i] = path_length

        if np.count_nonzero(feature1) == 0:
            return feature1

        min_path_length = np.min(feature1[feature1 != 0])

        for i in range(len(feature1)):
            if feature1[i] == 0:
                continue
            feature1[i] = min_path_length / feature1[i]

        if min_path_length > 15:
            feature1 *= 0.5
        elif min_path_length > 10:
            feature1 *= 0.75

        board_with_coins[current_position] = value_of_current_position

        return feature1

    def _feature_2(self, board_with_coins, current_position, next_positions):
        # collect coin with next move

        feature2 = np.zeros(len(self.actions))

        for i, next_position in enumerate(next_positions):
            if board_with_coins[next_position] == self.COIN_VALUE:
                feature2[i] = 1

        return feature2

    def _feature_3(self, board_without_explosion, current_position, bomb_is_possible):
        # invalid action

        feature3 = np.zeros(len(self.actions))

        for i, action in enumerate(self.actions):
            if action == 'WAIT':
                continue
            elif action == 'BOMB':
                if not bomb_is_possible:
                    feature3[i] = 1
            else:
                next_position = self._position_after_action(current_position, action)
                if board_without_explosion[next_position] != self.FREE_VALUE:
                    feature3[i] = 1

        return feature3

    def _feature_4(self, board, current_position, next_positions):
        # move out of explosion area

        feature4 = np.zeros(len(self.actions))

        if board[current_position] not in (self.BOMB_VALUE, self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE):
            return feature4

        path_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE))

        if path_out_of_explosion is not None:
            best_next_positions = [path[1] for path in path_out_of_explosion]
            for i, next_position in enumerate(next_positions):
                if next_position in best_next_positions:
                    feature4[i] = 1

        return feature4

    def _feature_5_old(self, board, current_position, next_positions):
        # move/stay in explosion area

        feature5 = np.zeros(len(self.actions))

        if board[current_position] == self.FREE_VALUE:
            for i, next_position in enumerate(next_positions):
                if board[next_position] in (self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE):
                    feature5[i] = 1
            return feature5

        paths_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE))

        if paths_out_of_explosion is not None:
            best_next_positions = [path[1] for path in paths_out_of_explosion]
            for i, next_position in enumerate(next_positions):
                if next_position not in best_next_positions:
                    feature5[i] = 1

        return feature5

    def _feature_5(self, board, explosion_timer_map, current_position, next_positions):
        # move/stay in explosion area

        # bomb timer     feature
        #     4            0.25
        #     3            0.5
        #     2            0.75
        #     1            1.0
        feature5 = np.zeros(len(self.actions))

        if board[current_position] == self.FREE_VALUE:
            for i, next_position in enumerate(next_positions):
                if board[next_position] in (self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE):
                    timer = explosion_timer_map[next_position]
                    if timer == 0:
                        continue
                    feature5[i] = (4 - explosion_timer_map[next_position]) / 3.
            return feature5

        paths_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE))

        if paths_out_of_explosion is not None:
            best_next_positions = [path[1] for path in paths_out_of_explosion]
            for i, next_position in enumerate(next_positions):
                if next_position not in best_next_positions:
                    min_bomb_timer = min(explosion_timer_map[current_position], explosion_timer_map.get(next_position, 3))
                    if min_bomb_timer == 0:
                        continue
                    feature5[i] = (4 - min_bomb_timer) / 3.

        return feature5

    def _feature_6_old(self, board_without_explosion, current_position, next_positions):
        # move towards nearest crate

        feature6 = np.zeros(len(self.actions))

        shortest_path_crates = self._shortest_path(board_without_explosion, current_position, self.CRATE_VALUE, free_tiles=(self.FREE_VALUE, self.CRATE_VALUE))

        if shortest_path_crates is not None:
            if len(shortest_path_crates[0]) == 2:
                # len==2 => standing next to crate
                return feature6

            best_next_positions = [path[1] for path in shortest_path_crates]
            for i, next_position in enumerate(next_positions):
                if next_position in best_next_positions:
                    feature6[i] = 1

        return feature6

    def _feature_6(self, board_without_explosion, current_position, next_positions):
        # steps to next crate for each direction

        feature6 = np.zeros(len(self.actions))

        for next_position in next_positions:
            if board_without_explosion[next_position] == self.CRATE_VALUE:
                # already standing next to crate
                return np.zeros(len(self.actions))

        value_of_current_position = board_without_explosion[current_position]
        board_without_explosion[current_position] = 10      # cannot go back to this tile

        for i, next_position in enumerate(next_positions):
            if board_without_explosion[next_position] != self.FREE_VALUE:
                continue

            shortest_path_crates = self._shortest_path(board_without_explosion, next_position, self.CRATE_VALUE, free_tiles=(self.FREE_VALUE, self.CRATE_VALUE))

            if shortest_path_crates is None:
                continue

            if len(shortest_path_crates[0]) == 2:
                # len==2 => standing next to crate
                feature6[i] = 1
            else:
                feature6[i] = len(shortest_path_crates[0]) - 1

        if np.count_nonzero(feature6) == 0:
            return feature6

        min_path_length = np.min(feature6[feature6 != 0])

        for i in range(len(feature6)):
            if feature6[i] == 0:
                continue
            feature6[i] = min_path_length / feature6[i]

        board_without_explosion[current_position] = value_of_current_position

        return feature6

    def _feature_7(self, board, current_position, bomb_is_possible):
        # bomb is suicidal (cannot escape)

        feature7 = np.zeros(len(self.actions))

        if not bomb_is_possible:
            return feature7

        path_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE)

        if path_out_of_explosion is not None:
            return feature7

        feature7[self.action_indices['BOMB']] = 1

        return feature7

    def _feature_8(self, board, current_position, bomb_is_possible, hit_crates):
        # bomb will destroy crates

        feature8 = np.zeros(len(self.actions))

        if not bomb_is_possible or not hit_crates:
            return feature8

        """
        path_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE)

        if path_out_of_explosion is None:
            # path to safe spot doesn't exists
            return feature8
        """

        feature8[self.action_indices['BOMB']] = 1
        return feature8

    def _feature_9(self, board_with_own_bomb, current_position, next_positions):
        # walk into dead_end (only if previous action was 'BOMB')
        return np.zeros(len(self.actions))

        feature9 = np.zeros(len(self.actions))

        if self.previous_action != 'BOMB':
            return feature9

        for i, next_position in enumerate(next_positions):
            if self.actions[i] == 'WAIT' or self.actions[i] == 'BOMB':
                continue

            if board_with_own_bomb[next_position] != self.OWN_EXPLOSION_VALUE:
                continue

            path_out_of_explosion = self._shortest_path(board_with_own_bomb, next_position, self.FREE_VALUE)

            if path_out_of_explosion is None:
                feature9[i] = 1

        return feature9

    def _feature_10_old(self, board_without_explosion, current_position, next_positions):
        # move towards nearest enemy

        feature10 = np.zeros(len(self.actions))

        shortest_path_enemy = self._shortest_path(board_without_explosion, current_position, self.ENEMY_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_VALUE))

        if shortest_path_enemy is not None:
            if len(shortest_path_enemy[0]) == 2:
                # len==2 => standing next to enemy
                return feature10

            best_next_positions = [path[1] for path in shortest_path_enemy]
            for i, next_position in enumerate(next_positions):
                if next_position in best_next_positions:
                    feature10[i] = 1

        return feature10

    def _feature_10(self, board_without_explosion, current_position, next_positions):
        # steps to next enemy for each direction

        feature10 = np.zeros(len(self.actions))

        number_of_crates = (board_without_explosion == self.CRATE_VALUE).sum()
        if number_of_crates > 5:
            return feature10

        for next_position in next_positions:
            if board_without_explosion[next_position] == self.ENEMY_VALUE:
                # already standing next to enemy
                return np.zeros(len(self.actions))

        value_of_current_position = board_without_explosion[current_position]
        board_without_explosion[current_position] = 10      # cannot go back to this tile

        for i, next_position in enumerate(next_positions):
            if board_without_explosion[next_position] != self.FREE_VALUE:
                continue

            shortest_path_enemies = self._shortest_path(board_without_explosion, next_position, self.ENEMY_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_VALUE))

            if shortest_path_enemies is None:
                continue

            if len(shortest_path_enemies[0]) == 2:
                # len==2 => standing next to enemy
                feature10[i] = 1
            else:
                feature10[i] = len(shortest_path_enemies[0]) - 1

        if np.count_nonzero(feature10) == 0:
            return feature10

        min_path_length = np.min(feature10[feature10 != 0])

        for i in range(len(feature10)):
            if feature10[i] == 0:
                continue
            feature10[i] = min_path_length / feature10[i]

        if min_path_length > 15:
            feature10 *= 0.5
        elif min_path_length > 10:
            feature10 *= 0.75

        board_without_explosion[current_position] = value_of_current_position

        return feature10

    def _feature_11(self, board, current_position, bomb_is_possible, next_positions):
        # place bomb if standing next to enemy

        feature11 = np.zeros(len(self.actions))

        if not bomb_is_possible:
            return feature11

        standing_next_to_enemy = False
        for i, next_position in enumerate(next_positions):
            if self.actions[i] == 'WAIT' or self.actions[i] == 'BOMB':
                continue

            if board[next_position] == self.ENEMY_VALUE:
                standing_next_to_enemy = True
                break

        if not standing_next_to_enemy:
            return feature11

        """
        path_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE)

        if path_out_of_explosion is None:
            # path to safe spot doesn't exists
            return feature11
        """

        feature11[self.action_indices['BOMB']] = 1
        return feature11

    def _feature_12(self, board, current_position, bomb_is_possible, enemy_inside_explosion_area):
        # place bomb if enemy is in explosion area

        feature12 = np.zeros(len(self.actions))

        if not bomb_is_possible or not enemy_inside_explosion_area:
            return feature12

        """
        path_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE)

        if path_out_of_explosion is None:
            # path to safe spot doesn't exists
            return feature12
        """

        feature12[self.action_indices['BOMB']] = 1
        return feature12

    def _feature_13(self, board_with_dead_ends, enemies, current_position, bomb_is_possible, enemy_inside_explosion_area):
        # place bomb if enemy is in explosion area and in dead end
        return np.zeros(len(self.actions))

        feature13 = np.zeros(len(self.actions))

        if not bomb_is_possible or not enemy_inside_explosion_area:
            return feature13

        direction_to_enemies = [[enemy[0] - current_position[0], enemy[1] - current_position[1]] for enemy in enemies]

        for i, direction in enumerate(direction_to_enemies):
            if direction.count(0) != 1:
                continue
            if direction[abs(1 - direction.index(0))] > settings.BOMB_POWER:
                continue

            if board_with_dead_ends[enemies[i]] == self.DEAD_END_VALUE:
                feature13[self.action_indices['BOMB']] = 1
                break

        return feature13

    def _feature_14(self, board_with_dead_ends, board_without_explosion, current_position, next_positions):
        # go into dead end if enemy is nearby

        feature14 = np.zeros(len(self.actions))

        shortest_path_enemy = self._shortest_path(board_without_explosion, current_position, self.ENEMY_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_VALUE))

        if shortest_path_enemy is None:
            return feature14

        path_length = len(shortest_path_enemy[0])
        if path_length > 4:
            return feature14

        for i, next_position in enumerate(next_positions):
            if board_with_dead_ends[next_position] == self.DEAD_END_VALUE:
                feature14[i] = 1

        return feature14

    def _shortest_path(self, board, start, target, free_tiles=(0, 5)):
        nearest_targets = []

        parents = {}
        parents[start] = start

        q = deque()
        q.append((start, 0))

        min_path_length = 1000

        while len(q) > 0:
            node, path_length = q.popleft()
            if path_length > min_path_length:
                continue

            if board[node] == target:
                nearest_targets.append(node)
                min_path_length = path_length

            x, y = node
            neighbors = [(nx, ny) for nx, ny in [(x, y-1), (x+1, y), (x, y+1), (x-1, y)] if board[(nx, ny)] in free_tiles]
            for neighbor in neighbors:
                if neighbor not in parents:
                    parents[neighbor] = node
                    q.append((neighbor, path_length + 1))

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

    def _get_dead_ends(self, board):
        dead_ends = [(x, y) for x in range(1, 16) for y in range(1, 16)
                     if board[x, y] == self.FREE_VALUE
                     and [board[x, y - 1], board[x + 1, y], board[x, y + 1], board[x - 1, y]].count(0) == 1]

        neighbor_indices = np.array([np.array([0, -1]), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])])

        board_with_dead_ends = board.copy()

        for dead_end in dead_ends:
            board_with_dead_ends[dead_end] = self.DEAD_END_VALUE

            length = 1
            direction = None
            x, y = dead_end
            neighbors = [board[x, y - 1], board[x + 1, y], board[x, y + 1], board[x - 1, y]]
            while neighbors.count(0) <= 2 and length <= 4:
                index = neighbors.index(0)
                direction = index
                x, y = [x, y] + neighbor_indices[index]
                neighbors = [board[x, y - 1], board[x + 1, y], board[x, y + 1], board[x - 1, y]]
                if neighbors.index(0) != direction:
                    break
                if neighbors.count(0) > 2:
                    break
                board_with_dead_ends[x, y] = self.DEAD_END_VALUE
                length += 1

        return board_with_dead_ends

    def _position_after_action(self, pos, action):
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

    def _action_is_valid(self, game_state: dict, action):
        if action == 'BOMB':
            return game_state['self'][2]
        elif action == 'WAIT':
            return True
        else:
            board = game_state['field'].copy()
            for enemy in game_state['others']:
                board[enemy[3]] = self.ENEMY_VALUE
            for bomb in game_state['bombs']:
                board[bomb[0]] = self.BOMB_VALUE

            if board[self._position_after_action(game_state['self'][3], action)] != self.FREE_VALUE:
                return False
            return True

    """
    def _observation(self, state: dict) -> np.ndarray:
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

        # walls and crates are given by the 'field' in the state
        walls = (state["field"] == -1).astype(float)
        crates = (state["field"] == 1).astype(float)

        # bombs and coins are individual positions
        bombs = np.zeros((s.COLS, s.ROWS), dtype=float)
        coins = np.zeros((s.COLS, s.ROWS), dtype=float)

        for b in state["bombs"]:
            bombs[b[0][0], b[0][1]] = (5. - b[1]) / 4.

        for c in state["coins"]:
            coins[c[0], c[1]] = 1.0

        player = np.zeros((s.COLS, s.ROWS), dtype=float)

        p = state["self"][3]

        player[p[0], p[1]] = 1.0 if state["self"][2] else -1.0

        enemies = np.zeros((s.COLS, s.ROWS), dtype=float)

        for o in state["others"]:
            op = o[3]
            enemies[op[0], op[1]] = 1.0

        observation = np.stack([
            walls,
            crates,
            bombs,
            coins,
            player,
            enemies
        ], axis=-1)

        return np.expand_dims(observation, axis=0)
    """
