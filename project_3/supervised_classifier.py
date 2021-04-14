import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pathlib


class SupervisedClassifier:
    def __init__(self, args, num_classes, save_path, input_shape):
        self.args = args
        self.num_classes = num_classes

        self.save_path = save_path.joinpath("supervised_classifier")
        self.save_path.mkdir(exist_ok=True)

        self.input_shape = input_shape
        self.history_dict = dict()

        self.build_models()

    def build_models(self):
        self.model = keras.Sequential(
            [
                layers.Conv2D(32, input_shape=self.input_shape, kernel_size=3,
                              padding="same", activation="relu",),
                layers.MaxPooling2D(pool_size=2),
                layers.Conv2D(64, kernel_size=3, padding="same",
                              activation="relu"),
                layers.MaxPooling2D(pool_size=2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(10, activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ],
            name="supervised_classifier"
        )

        self.model.summary()

        optimizer = keras.optimizers.get(
            self.args.optimizer_supervised_classifier)
        if self.args.learning_rate_supervised_classifier is not None:
            optimizer.learning_rate.assign(
                self.args.learning_rate_supervised_classifier)

        self.model.compile(
            optimizer=optimizer,
            loss=self.args.loss_function_supervised_classifier,
            metrics=["accuracy"])

    def train(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(x_train, y_train,
                                 batch_size=self.args.batch_size,
                                 epochs=self.args.epochs_supervised_classifier,
                                 validation_data=(x_val, y_val))

        self.history_dict = history.history

        return self.history_dict

    def save_models(self):
        self.model.save(self.save_path.joinpath("model"))
        np.save(self.save_path.joinpath(
            "supervised_classifier_history_dict.npy"), self.history_dict)

    def load_models(self):
        self.model = keras.models.load_model(
            self.save_path.joinpath("model"))
        self.history_dict = np.load(self.save_path.joinpath(
            "supervised_classifier_history_dict.npy"), allow_pickle=True).item()

    def evaluate(self, x_test, y_test, verbose=1):
        return self.model.evaluate(x_test, y_test, verbose=verbose)
