import tensorflow as tf
import numpy as np

import settings as s

from ._core import DQBase
from ._diagnostics import Diagnostics
from .memory import Memory
from .. import utils


class DQTrain(DQBase):
    def __init__(self, actions, epsilon=0.4, gamma=0.9, learning_rate=1e-3):
        model = tf.keras.Sequential([
            # apparently, a convolutional layer combined with a maxpool layer
            # can help first identify features, and then filter for the most interesting features
            tf.keras.layers.Conv2D(filters=36, kernel_size=4, strides=1, input_shape=(9, 9, 6),  # (s.COLS, s.ROWS, 6),
                                   activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            # do this again but finer
            tf.keras.layers.Conv2D(filters=36, kernel_size=2, strides=1, activation="relu"), #input_shape=(9, 9, 6)),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
            tf.keras.layers.Flatten(),  # flatten input for dense layers
            tf.keras.layers.Dense(units=64, activation="relu"),  # hidden layer to process features
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=len(actions), activation=None)  # output layer
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = tf.keras.losses.MeanSquaredError()
        metrics = [
            tf.keras.metrics.Accuracy()
        ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.epsilon = epsilon
        self.gamma = gamma

        self.memory = Memory()
        self.diagnostics = Diagnostics(actions)

        super().__init__(actions, model)

    def act(self, state: dict) -> str:
        if utils.random(self.epsilon):
            action = self.explore(state)
            return self.actions[action]

        return super().act(state)

    def explore(self, state: dict) -> [np.ndarray, int]:
        return np.random.choice(np.arange(len(self.actions)))

    def memorize(self, state, action, new_state, events):
        observation = DQTrain.observation(state)
        logits = self.model.predict(observation)
        prediction = tf.squeeze(tf.nn.softmax(logits))
        action = np.argmax(self.actions == action)
        reward = self.reward(events)

        if not self.memory.episodes:
            self.memory.add_episode()
            self.diagnostics.last_episode_predictions.clear()

        self.diagnostics.predictions.append(prediction)
        self.diagnostics.last_episode_predictions.append(prediction)

        self.memory.add(observation, action, reward)

    def finalize(self):
        r = self.memory.finalize(self.gamma)
        self.memory.aggregate()

        self.epsilon *= 0.99

        self.diagnostics.returns.append(r)
        self.diagnostics.epsilons.append(self.epsilon)

    def train(self, epochs=1, batch_size=None, reduce: int = None, clear_memory=False):
        for epoch in range(epochs):
            if batch_size is None:
                memory = self.memory
            else:
                memory = self.memory.random_batch(batch_size)

            with tf.GradientTape() as tape:
                logits = self.model(np.vstack(memory.observations), training=True)
                predictions = tf.nn.softmax(logits, axis=-1)

                indices = tf.expand_dims(memory.actions, axis=1)

                # y_pred is the predicted return for action a in state s, i.e. Q(s, a)
                # y_true is the return that was actually seen in a previous episode for action a in state s
                y_pred = tf.gather(predictions, indices, axis=1)
                y_true = tf.nn.sigmoid(memory.returns)

                loss = self.model.compiled_loss(y_true, y_pred)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        if reduce is not None:
            self.memory.reduce(reduce)

        if clear_memory:
            self.memory.clear()

    from ._reward import reward
    reward = staticmethod(reward)
