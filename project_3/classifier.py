import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pathlib


class Classifier:
    class ClassifierModel(keras.Model):
        def __init__(self, encoder, classifier_head):
            super().__init__()
            self.encoder = encoder
            self.classifier_head = classifier_head

        def call(self, x):
            encoded = self.encoder(x)
            classified = self.classifier_head(encoded)
            return classified

    def __init__(self, args, num_classes, encoder_to_copy, save_path):
        self.args = args
        self.num_classes = num_classes

        self.encoder = keras.models.clone_model(encoder_to_copy)
        if args.freeze:
            for layer in self.encoder.layers:
                layer.trainable = False
        self.encoder.set_weights(encoder_to_copy.get_weights())

        self.save_path = save_path.joinpath("classifier")
        self.save_path.mkdir(exist_ok=True)

        self.history_dict = dict()

        self.build_models()

    def build_models(self):
        self.classifier_head = keras.Sequential(
            [
                layers.Dense(10, input_shape=(
                    self.args.latent_vector_size, ), activation="relu"),
                layers.Dense(self.num_classes, activation="softmax"),
            ],
            name="classifier_head"
        )

        self.classifier_head.summary()

        self.classifier_model = self.ClassifierModel(
            self.encoder, self.classifier_head)

    def train(self, x_train, y_train, x_val, y_val):
        # TODO map constants to keras stuff
        self.classifier_model.compile(loss=self.args.loss_function_classifier,
                                      optimizer=self.args.optimizer_classifier,
                                      metrics=["accuracy"])

        if(self.args.learning_rate_classifier is not None):
            keras.backend.set_value(
                self.classifier_model.optimizer.learning_rate, self.args.learning_rate_classifier)

        history = self.classifier_model.fit(x_train, y_train,
                                            batch_size=self.args.batch_size,
                                            epochs=self.args.epochs_classifier,
                                            validation_data=(x_val, y_val))

        self.history_dict = history.history

        return self.history_dict

    def save_models(self):
        self.encoder.save(self.save_path.joinpath("encoder_in_classifier"))
        self.classifier_head.save(self.save_path.joinpath("classifier_head"))
        self.classifier_model.save(self.save_path.joinpath("classifer"))
        np.save(self.save_path.joinpath(
            "classifier_history_dict.npy"), self.history_dict)

    def load_models(self):
        self.encoder = keras.models.load_model(
            self.save_path.joinpath("encoder_in_classifier"))
        self.classifier_head = keras.models.load_model(
            self.save_path.joinpath("classifier_head"))
        self.classifier_model = keras.models.load_model(
            self.save_path.joinpath("classifer"))
        self.history_dict = np.load(self.save_path.joinpath(
            "classifier_history_dict.npy"), allow_pickle=True).item()

    def evaluate(self, x_test, y_test, verbose=1):
        return self.classifier_model.evaluate(x_test, y_test, verbose=verbose)
