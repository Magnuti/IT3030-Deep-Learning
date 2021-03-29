import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pathlib


class SupervisedClassifier:
    class ClassifierModel(keras.Model):
        def __init__(self, classifier_body, classifier_head):
            super().__init__()
            self.classifier_body = classifier_body
            self.classifier_head = classifier_head

        def call(self, x):
            x = self.classifier_body(x)
            classified = self.classifier_head(x)
            return classified

    def __init__(self, args, num_classes, save_path):
        self.args = args
        self.num_classes = num_classes

        self.save_path = save_path.joinpath("supervised_classifier")
        self.save_path.mkdir(exist_ok=True)

        self.history_dict = dict()

        self.build_models()

    def build_models(self):
        self.classifier_body = keras.Sequential(
            [
                layers.Conv2D(32, input_shape=(28, 28, 1),  # TODO fix
                              kernel_size=3, padding="same", activation="relu",),
                layers.MaxPooling2D(pool_size=2),
                layers.Conv2D(64, kernel_size=3, padding="same",
                              activation="relu"),
                layers.MaxPooling2D(pool_size=2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.args.latent_vector_size),  # TODO af here?
            ],
            name="classifier_body"
        )

        # ! Just copied from classifier.py lol
        self.classifier_head = keras.Sequential(
            [
                # TODO af param here?
                # TODO take in neuron count param
                layers.Dense(10, input_shape=(self.args.latent_vector_size, )),
                layers.Dense(self.num_classes, activation="softmax"),
            ],
            name="classifier_head"
        )

        self.classifier_body.summary()

        self.classifier_model = self.ClassifierModel(
            self.classifier_body, self.classifier_head)

    def train(self, x_train, y_train):
        # TODO map constants to keras stuff
        self.classifier_model.compile(loss="categorical_crossentropy",
                                      optimizer="adam", metrics=["accuracy"])

        history = self.classifier_model.fit(x_train, y_train, batch_size=self.args.batch_size,
                                            epochs=self.args.epochs_classifier, validation_split=0.1)

        self.history_dict = history.history

        return self.history_dict

    def save_models(self):
        self.classifier_body.save(self.save_path.joinpath("classifier_body"))
        self.classifier_head.save(self.save_path.joinpath("classifier_head"))
        np.save(self.save_path.joinpath(
            "supervised_classifier_history_dict.npy"), self.history_dict)

    def load_models(self):
        self.classifier_body = keras.models.load_model(
            self.save_path.joinpath("classifier_body"))
        self.classifier_head = keras.models.load_model(
            self.save_path.joinpath("classifier_head"))
        self.history_dict = np.load(self.save_path.joinpath(
            "supervised_classifier_history_dict.npy"), allow_pickle=True).item()

        self.classifier_model = self.ClassifierModel(
            self.classifier_body, self.classifier_head)

    # def evaluate(self, x_test, y_test):
    #     score = self.model.evaluate(x_test, y_test, verbose=0)
    #     print("Test loss:", score[0])
    #     print("Test accuracy:", score[1])
