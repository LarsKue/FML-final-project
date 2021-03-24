
import tensorflow as tf
import numpy as np

from typing import Tuple

import events as e
import settings as s

from .dqbase import DQBase
from ..memory import Memory
from ...utils import *


class DQTrainer(DQBase):
    def __init__(self, actions: np.ndarray, epsilon=0.5, temperature=1.0, gamma=0.9):
        model = tf.keras.Sequential([
            # apparently, a convolutional layer combined with a maxpool layer
            # can help first identify features, and then filter for the most interesting features
            tf.keras.layers.Conv2D(filters=36, kernel_size=5, strides=(1, 1), input_shape=(s.COLS, s.ROWS, 6), activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # do this again but finer
            tf.keras.layers.Conv2D(filters=36, kernel_size=3, strides=(1, 1), activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),  # flatten input for dense layers
            tf.keras.layers.Dense(units=64, activation="relu"),  # hidden layer to process features
            tf.keras.layers.Dense(units=len(actions), activation=None)  # output layer
        ])

        self.epsilon = epsilon
        self.temperature = temperature
        self.discount = gamma

        self.memory = Memory()

        self.feature_action_counts = {}

        super().__init__(actions, model)

    def to_base(self) -> DQBase:
        return DQBase(self.actions, self.model)

    def action_counts(self, prediction):
        # use neural net predictions as quasi-states
        # each action gets 5 bins, leading to 5^6 ~ 15k possible states
        # meaning we can expect a relevant effect from this method after ~150k states have been seen
        # which should happen after at most 30k episodes
        n_bins_per_action = 5
        bins = np.linspace(0, 1, n_bins_per_action)
        binned_prediction, _ = np.histogram(prediction, bins)
        h = tuple(binned_prediction)

        action_counts = self.feature_action_counts.get(h)

        if action_counts is None:
            # have not been in this state yet
            action_counts = np.zeros(self.actions.shape, dtype=int)
            self.feature_action_counts[h] = action_counts

        return action_counts

    def act(self, state: dict) -> Tuple[str, np.ndarray]:
        if rand(self.epsilon):
            observation, prediction, action = self.explore(state)
        else:
            observation, prediction, action = super().exploit(state)

        return self.actions[action], prediction

    def explore(self, state: dict) -> [np.ndarray, int]:
        observation = super().observation(state)

        _, prediction = self.predict(observation)

        action = np.random.choice(np.arange(len(self.actions)))

        return observation, prediction, action

    # def explore(self, state: dict) -> [np.ndarray, np.ndarray, int]:
    #     """
    #     Explore the game world outside of the current policy
    #     """
    #     observation = super().observation(state)
    #
    #     _, prediction = self.predict(observation)
    #
    #     action_counts = self.action_counts(prediction)
    #
    #     # prefer actions we have not tried much, randomized for ties
    #     # values = softmin(action_counts / self.temperature)
    #     values = tf.nn.softmax(-action_counts / self.temperature)
    #
    #     action = np.random.choice(np.arange(len(self.actions)), p=values)
    #
    #     return observation, prediction, action

    # def train(self, optimizer, steps, batch_size=64, reduce=None, clear_memory=True):
    #     discounted_rewards = discount(self.memory.rewards[-steps:], self.discount)
    #
    #     self.memory.returns.extend(discounted_rewards)
    #
    #     memory = self.memory.random_batch(batch_size)
    #
    #     with tf.GradientTape() as tape:
    #         # Forward Propagation
    #         logits = self.model(np.vstack(memory.observations))
    #
    #         loss = find_loss(logits, np.array(memory.actions), np.array(memory.returns))
    #
    #     # Back Propagation
    #     if batch_size <= len(memory):
    #         # only train the net if the memory is sufficently large
    #         # only loss gradient is determining (since loss may be negative)
    #         gradients = tape.gradient(loss, self.model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    #
    #     min_epsilon = 0.15
    #     if self.epsilon > min_epsilon:
    #         sink_rate = 0.01
    #         self.epsilon = self.epsilon * (1 / (1 + sink_rate * self.epsilon))
    #     else:
    #         self.epsilon = min_epsilon
    #
    #     if reduce is not None:
    #         self.memory.reduce(reduce)
    #
    #     if clear_memory:
    #         self.memory.clear()
    #
    #     return loss, np.sum(discounted_rewards)

    def __reward(self, events):
        # score for any non-listed move
        other = 0

        # event_rewards = {
        #     e.INVALID_ACTION: -6,
        #     e.WAITED: -6,
        #     e.CRATE_DESTROYED: 5,
        #     e.BOMB_DROPPED: 0.1,
        #     e.GOT_KILLED: -35,
        #     e.KILLED_SELF: 5,
        #     e.SURVIVED_ROUND: 5,
        #     e.COIN_COLLECTED: 30,
        #     e.KILLED_OPPONENT: 150,
        # }

        # event_rewards = {
        #     e.INVALID_ACTION: -4,
        #     e.MOVED_UP: 0,
        #     e.MOVED_DOWN: 0,
        #     e.MOVED_LEFT: 0.1,
        #     e.MOVED_RIGHT: 0.1,
        #     e.GOT_KILLED: -50,
        #     e.COIN_COLLECTED: 100,
        #     e.KILLED_OPPONENT: 150,
        # }

        event_rewards = {
            e.INVALID_ACTION: -200,
            e.CRATE_DESTROYED: 10,
            e.COIN_FOUND: 0,
            e.COIN_COLLECTED: 150,
            e.KILLED_OPPONENT: 500,
            e.KILLED_SELF: 0,
            e.GOT_KILLED: -1000,
        }

        rewards = list(map(event_rewards.get, events))
        rewards = [r if r is not None else other for r in rewards]
        reward = np.sum(rewards)

        return reward / 100

    def reward(self, state, action, new_state, events):
        observation = self.observation(state)
        action = np.argmax(self.actions == action)
        _, prediction = self.predict(observation)
        reward = self.__reward(events)
        if new_state is None:
            new_observation = None
        else:
            new_observation = self.observation(new_state)

        if not self.memory.episodes:
            self.memory.add_episode()

        self.memory.add(observation, action, reward, new_observation)

        # increment action counter
        self.action_counts(prediction)[action] += 1

    def train(self, optimizer, gamma, epsilon_updater, batch_size=None, reduce_memory=None, clear_memory=False):
        reward = self.memory.finalize(gamma)
        self.memory.aggregate()

        self.epsilon = epsilon_updater(self.epsilon)

        if batch_size is not None:
            if batch_size > len(self.memory):
                # only train if we have sufficient batches available
                return None, reward

            memory = self.memory.random_batch(batch_size)
        else:
            # no random batching
            memory = self.memory

        with tf.GradientTape() as tape:
            # Forward Propagation
            logits = self.model(np.vstack(memory.states))

            loss = find_loss(logits, np.array(memory.actions), np.array(memory.returns))

        # Back Propagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if reduce_memory is not None:
            self.memory.reduce(reduce_memory)

        if clear_memory:
            self.memory.clear()

        return loss, reward
