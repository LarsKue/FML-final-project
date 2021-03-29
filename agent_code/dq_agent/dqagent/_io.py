
from pathlib import Path
import _pickle as cpickle

import tensorflow as tf
import shutil


def load(path):
    # saving or loading agents is non-trivial since tf.keras.model is only weakly referenced
    # and thus cannot be directly dumped to a file

    with open(path / Path("agent"), "rb") as f:
        instance = cpickle.load(f)

    instance.model = tf.keras.models.load_model(path / Path("model"))

    return instance


def save(self, path):
    self.model.save(path / Path("model"))

    # temporarily remove the model, save the rest of the class and re-add the model
    temp = self.__dict__.pop("model")
    with open(path / Path("agent"), "wb+") as f:
        cpickle.dump(self, f)

    self.model = temp
