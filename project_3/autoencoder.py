import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses

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

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.build_models()
        self.autoencoder_model = self.AutoEncoderModel(
            self.encoder, self.decoder)

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
                layers.Conv2D(32, input_shape=self.encoder_input_shape,  kernel_size=(
                    3, 3), padding="same", activation="relu",),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3),
                              padding="same", activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
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
                layers.UpSampling2D(size=(2, 2)),
                layers.Conv2DTranspose(
                    32, 3, activation="relu", padding="same"),
                layers.UpSampling2D(size=(2, 2)),
                layers.Conv2DTranspose(
                    1, 3, activation="relu", padding="same"),
                # layers.Flatten(),
                # layers.Dense(self.input_shape[0] * self.input_shape[1])  # TODO
            ],
            name="decoder"
        )

        self.decoder.summary()

    def train(self, x_train):
        # TODO map constants to keras stuff
        # self.compile(loss="categorical_crossentropy",
        #              optimizer="adam", metrics=["accuracy"])
        self.autoencoder_model.compile(
            optimizer='adam', loss=losses.MeanSquaredError())

        self.autoencoder_model.fit(x_train, x_train, batch_size=self.args.batch_size,
                                   epochs=self.args.epochs_classifier, validation_split=0.1)

    def save_models(self):
        # TODO fix path
        self.encoder.save("saves/encoder")
        self.decoder.save("saves/decoder")

    def load_models(self):
        self.encoder = keras.models.load_model("saves/encoder")
        self.decoder = keras.models.load_model("saves/decoder")

    # def evaluate(self, x_test, y_test):
    #     score = self.model.evaluate(x_test, y_test, verbose=0)
    #     print("Test loss:", score[0])
    #     print("Test accuracy:", score[1])
