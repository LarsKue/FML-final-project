from typing import List
import numpy as np
from collections import deque

import events as e
import settings


class SorryDaveAgent:

    def __init__(self, train=False, weights=None, logger=None):
        self.num_features = 12
        if weights is None:
            self.weights = np.zeros(self.num_features)
        else:
            self.weights = weights

        self.train = train

        self.logger = logger

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

            self.experience_buffer = [[]]

            self.update_number = 0
            self.weights_updates = np.zeros((2001, self.num_features))
            self.weights_updates[0] = self.weights

    def act(self, game_state: dict):
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
        if self.logger is not None:
            self.logger.info(f"weights = {np.round(self.weights, 3)}")

    def _explore(self, game_state: dict):
        action = np.random.choice(self.actions)
        return action

    def _exploit(self, game_state: dict):
        features = self._get_features(game_state)

        q = self.weights.dot(features)
        best_actions = np.argwhere(q == np.amax(q)).flatten()

        action = self.actions[np.random.choice(best_actions)]

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
                reward += 0              # sd == KILLED_SELF + GOT_KILLED => reward(sd) = -1000
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
            explosion_timer_map[(bomb_x, bomb_y)] = min(timer, explosion_timer_map.get((bomb_x, bomb_y), 4))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x + i, bomb_y)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 4))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x - i, bomb_y)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 4))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x, bomb_y + i)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 4))

            for i in range(1, settings.BOMB_POWER + 1):
                coord = (bomb_x, bomb_y - i)
                if board[coord] == self.WALL_VALUE:
                    break
                if board[coord] == self.FREE_VALUE:
                    board[coord] = self.ENEMY_EXPLOSION_VALUE
                    explosion_timer_map[coord] = min(timer, explosion_timer_map.get(coord, 4))

        board_with_own_bomb = board.copy()
        board_only_own_bomb = board_without_explosion.copy()
        hit_crates = False
        enemy_inside_explosion_area = False
        if bomb_is_possible:
            # placing bomb is possible
            bomb_x, bomb_y = current_position
            board_with_own_bomb[current_position] = 3
            board_only_own_bomb[current_position] = 3
            explosion_timer_map[current_position] = min(4, explosion_timer_map.get(current_position, 4))

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
                    explosion_timer_map[coord] = min(4, explosion_timer_map.get(coord, 4))

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
                    explosion_timer_map[coord] = min(4, explosion_timer_map.get(coord, 4))

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
                    explosion_timer_map[coord] = min(4, explosion_timer_map.get(coord, 4))

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
                    explosion_timer_map[coord] = min(4, explosion_timer_map.get(coord, 4))

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
            # + move towards nearest enemy
            self._feature_9(board_without_explosion, current_position, next_positions),
            # + place bomb if standing next to enemy
            self._feature_10(board, current_position, bomb_is_possible, next_positions),
            # + place bomb if enemy is in explosion area
            self._feature_11(board_only_own_bomb, current_position, bomb_is_possible, enemy_inside_explosion_area),
            # - go into dead end if enemy is nearby
            self._feature_12(board_with_dead_ends, board_without_explosion, current_position, next_positions)
        ], axis=0)

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
                    feature5[i] = (5 - explosion_timer_map[next_position]) / 4.
            return feature5

        paths_out_of_explosion = self._shortest_path(board, current_position, self.FREE_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_EXPLOSION_VALUE, self.OWN_EXPLOSION_VALUE))

        if paths_out_of_explosion is not None:
            best_next_positions = [path[1] for path in paths_out_of_explosion]
            for i, next_position in enumerate(next_positions):
                if next_position not in best_next_positions:
                    min_bomb_timer = min(explosion_timer_map[current_position], explosion_timer_map.get(next_position, 4))
                    feature5[i] = (5 - min_bomb_timer) / 4.

        return feature5

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

        feature8[self.action_indices['BOMB']] = 1
        return feature8

    def _feature_9(self, board_without_explosion, current_position, next_positions):
        # steps to next enemy for each direction

        feature9 = np.zeros(len(self.actions))

        number_of_crates = (board_without_explosion == self.CRATE_VALUE).sum()
        if number_of_crates > 5:
            return feature9

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
                feature9[i] = 1
            else:
                feature9[i] = len(shortest_path_enemies[0]) - 1

        if np.count_nonzero(feature9) == 0:
            return feature9

        min_path_length = np.min(feature9[feature9 != 0])

        for i in range(len(feature9)):
            if feature9[i] == 0:
                continue
            feature9[i] = min_path_length / feature9[i]

        board_without_explosion[current_position] = value_of_current_position

        return feature9

    def _feature_10(self, board, current_position, bomb_is_possible, next_positions):
        # place bomb if standing next to enemy

        feature10 = np.zeros(len(self.actions))

        if not bomb_is_possible:
            return feature10

        standing_next_to_enemy = False
        for i, next_position in enumerate(next_positions):
            if self.actions[i] == 'WAIT' or self.actions[i] == 'BOMB':
                continue

            if board[next_position] == self.ENEMY_VALUE:
                standing_next_to_enemy = True
                break

        if not standing_next_to_enemy:
            return feature10

        feature10[self.action_indices['BOMB']] = 1
        return feature10

    def _feature_11(self, board, current_position, bomb_is_possible, enemy_inside_explosion_area):
        # place bomb if enemy is in explosion area

        feature11 = np.zeros(len(self.actions))

        if not bomb_is_possible or not enemy_inside_explosion_area:
            return feature11

        feature11[self.action_indices['BOMB']] = 1
        return feature11

    def _feature_12(self, board_with_dead_ends, board_without_explosion, current_position, next_positions):
        # go into dead end if enemy is nearby

        feature12 = np.zeros(len(self.actions))

        shortest_path_enemy = self._shortest_path(board_without_explosion, current_position, self.ENEMY_VALUE, free_tiles=(self.FREE_VALUE, self.ENEMY_VALUE))

        if shortest_path_enemy is None:
            return feature12

        path_length = len(shortest_path_enemy[0])
        if path_length > 4:
            return feature12

        for i, next_position in enumerate(next_positions):
            if board_with_dead_ends[next_position] == self.DEAD_END_VALUE:
                feature12[i] = 1

        return feature12

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
