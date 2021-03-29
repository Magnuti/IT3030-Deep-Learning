import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses
import pathlib

from constants import Dataset


class AutoEncoder:
    class AutoEncoderModel(keras.Model):

        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path.joinpath("autoencoder")
        self.save_path.mkdir(exist_ok=True)

        self.history_dict = dict()

        self.build_models()

    def build_models(self):
        if self.args.dataset == Dataset.MNIST:
            self.encoder_input_shape = (28, 28, 1)
        elif self.args.dataset == Dataset.FASHION_MNIST:
            self.encoder_input_shape = (28, 28, 1)
        else:
            raise NotImplementedError()

        print("Encoder input shape:", self.encoder_input_shape)
        self.encoder = keras.Sequential(
            [
                # keras.Input(shape=self.input_shape),
                layers.Conv2D(32, input_shape=self.encoder_input_shape,
                              kernel_size=3, padding="same", activation="relu",),
                layers.MaxPooling2D(pool_size=2),
                layers.Conv2D(64, kernel_size=3, padding="same",
                              activation="relu"),
                layers.MaxPooling2D(pool_size=2),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.args.latent_vector_size),  # TODO af here?
            ],
            name="encoder"
        )

        self.encoder.summary()

        decoder_input_shape = (self.args.latent_vector_size, )
        print("Decoder input size:", decoder_input_shape)

        self.decoder = keras.Sequential(
            [
                # layers.Input(shape=(4, 1)),
                layers.Dense(3136, input_shape=decoder_input_shape),  # TODO
                layers.Reshape((7, 7, 64)),
                layers.UpSampling2D(size=2),
                layers.Conv2DTranspose(
                    32, 3, activation="relu", padding="same"),
                layers.UpSampling2D(size=2),
                layers.Conv2DTranspose(
                    1, 3, activation="relu", padding="same"),
                # layers.Flatten(),
                # layers.Dense(self.input_shape[0] * self.input_shape[1])  # TODO
            ],
            name="decoder"
        )

        self.decoder.summary()

        self.autoencoder_model = self.AutoEncoderModel(
            self.encoder, self.decoder)

    def train(self, x_train, x_test):
        # TODO map constants to keras stuff
        # self.compile(loss="categorical_crossentropy",
        #              optimizer="adam", metrics=["accuracy"])
        # TODO fix loss param
        self.autoencoder_model.compile(optimizer='adam', metrics=[
                                       'accuracy'], loss=losses.MeanSquaredError())

        # TODO look over validation split
        history = self.autoencoder_model.fit(x_train, x_train, batch_size=self.args.batch_size,
                                             epochs=self.args.epochs_classifier, validation_split=0.1, validation_data=(x_test, x_test))
        self.history_dict = history.history

        return self.history_dict

    def save_models(self):
        self.encoder.save(self.save_path.joinpath("encoder"))
        self.decoder.save(self.save_path.joinpath("decoder"))
        np.save(self.save_path.joinpath(
            "autoencoder_history_dict.npy"), self.history_dict)

    def load_models(self):
        self.encoder = keras.models.load_model(
            self.save_path.joinpath("encoder"))
        self.decoder = keras.models.load_model(
            self.save_path.joinpath("decoder"))
        self.history_dict = np.load(self.save_path.joinpath(
            "autoencoder_history_dict.npy"), allow_pickle=True).item()

        self.autoencoder_model = self.AutoEncoderModel(
            self.encoder, self.decoder)

    # def evaluate(self, x_test, y_test):
    #     score = self.model.evaluate(x_test, y_test, verbose=0)
    #     print("Test loss:", score[0])
    #     print("Test accuracy:", score[1])
