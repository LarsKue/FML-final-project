import tensorflow as tf
import numpy as np

import settings as s

from ._core import DQBase
from ._diagnostics import Diagnostics
from .memory import Memory
from .. import utils


class DQTrain(DQBase):
    def __init__(self, actions, model=None, optimizer=None, loss=None, metrics=None, epsilon=1.0, gamma=0.95,
                 learning_rate=1e-4):
        if model is None:
            model = tf.keras.Sequential([
                # a convolutional layer combined with a maxpool layer
                # can help first identify features, and then filter for the most interesting features
                tf.keras.layers.Conv2D(filters=36, kernel_size=4, strides=1, input_shape=(s.COLS, s.ROWS, 6),
                                       activation="elu"),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
                # do this again but finer
                tf.keras.layers.Conv2D(filters=36, kernel_size=2, strides=1, activation="elu"),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
                tf.keras.layers.Flatten(),  # flatten input for dense layers
                tf.keras.layers.Dense(units=512, activation="elu"),  # hidden layers to process features
                tf.keras.layers.Dense(units=64, activation="elu"),
                # tf.keras.layers.Dropout(rate=0.2),  # uncomment if necessary
                tf.keras.layers.Dense(units=len(actions), activation=None)  # output layer
            ])

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if loss is None:
            loss = tf.keras.losses.MeanSquaredError()

        if metrics is None:
            metrics = [
                tf.keras.metrics.Accuracy()
            ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.epsilon = epsilon
        self.gamma = gamma

        self.memory = Memory()
        self.diagnostics = Diagnostics()

        super().__init__(actions, model)

    def act(self, state: dict) -> str:
        if utils.random(self.epsilon):
            action = self.explore(state)
            return self.actions[action]

        return super().act(state)

    def explore(self, state: dict) -> int:
        # randomly choose an action that the current policy deems poor
        # this ensures that options which the agent does not usually
        # consider are chosen more often, thus leading to exploration
        observation = DQTrain.observation(state)
        logits = self.model.predict(observation)
        prediction = tf.nn.softmax(logits)
        # must be float64 or else numpy complains that the probabilities don't sum to 1
        prediction = tf.cast(1 - prediction, dtype=tf.float64)
        prediction /= tf.math.reduce_sum(prediction, axis=-1)
        prediction = tf.squeeze(prediction)

        return np.random.choice(np.arange(len(self.actions)), p=prediction)

    def memorize(self, state, action, new_state, events):
        observation = DQTrain.observation(state)
        action = np.argmax(self.actions == action)
        reward = self.reward(events)
        new_observation = DQTrain.observation(new_state)

        self.memory.add(observation, action, reward, new_observation)

    def finalize(self):
        self.memory.finalize(self.gamma)
        self.memory.aggregate()

    def train(self, epochs=1, batch_size=None, reduce: int = None, clear_memory=False):
        for epoch in range(epochs):
            if batch_size is None:
                memory = self.memory
            else:
                memory = self.memory.random_batch(batch_size)

            with tf.GradientTape() as tape:
                logits = self.model(np.vstack(memory.observations), training=True)
                indices = tf.convert_to_tensor(memory.actions)
                y_pred = tf.linalg.tensor_diag_part(tf.gather(logits, indices, axis=1))

                new_logits = self.model(np.vstack(memory.new_observations), training=True)
                rewards = tf.convert_to_tensor(memory.rewards, dtype=tf.float32)
                new_indices = tf.argmax(new_logits, axis=-1)
                y_true = rewards + self.gamma * tf.linalg.tensor_diag_part(tf.gather(new_logits, new_indices, axis=1))

                # L = MSE(y_true, Q(s, a)), where y_true = r + gamma * max_a' Q(s', a')
                # s' is the state after s
                loss = self.model.compiled_loss(y_true, y_pred)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if reduce is not None:
            self.memory.reduce(reduce)

        if clear_memory:
            self.memory.clear()

    from ._reward import reward as _reward
    reward = _reward
