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

    def __init__(self, args, save_path, input_shape):
        super().__init__()
        self.args = args
        self.save_path = save_path.joinpath("autoencoder")
        self.save_path.mkdir(exist_ok=True)
        self.input_shape = input_shape

        self.history_dict = dict()

        self.build_models()

    def build_models(self):
        print("Encoder input shape:", self.input_shape)
        self.encoder = keras.Sequential(
            [
                # keras.Input(shape=self.input_shape),
                layers.Conv2D(32, input_shape=self.input_shape,
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

        # Makes sure the output image is the same dimension as the input image
        # Hotfix for handling 28x28 and 32x32 images
        # Multiply and divide by 4 since the convolutions increase/decrease the original image by a factor of 4
        neurons_in_first_decoder_layers = self.input_shape[0] * \
            self.input_shape[1] * 4
        reshape_number = self.input_shape[0] // 4
        self.decoder = keras.Sequential(
            [
                # layers.Input(shape=(4, 1)),
                layers.Dense(neurons_in_first_decoder_layers,
                             input_shape=decoder_input_shape),
                layers.Reshape(
                    (reshape_number, reshape_number, self.args.batch_size)),
                layers.UpSampling2D(size=2),
                layers.Conv2DTranspose(
                    32, 3, activation="relu", padding="same"),
                layers.UpSampling2D(size=2),
                layers.Conv2DTranspose(
                    self.input_shape[-1], 3, activation="relu", padding="same"),
            ],
            name="decoder"
        )

        self.decoder.summary()

        self.autoencoder_model = self.AutoEncoderModel(
            self.encoder, self.decoder)

    def train(self, x_train, x_val):
        self.autoencoder_model.compile(
            optimizer=self.args.optimizer_autoencoder,
            loss=self.args.loss_function_auto_encoder)

        if(self.args.learning_rate_auto_encoder is not None):
            keras.backend.set_value(
                self.autoencoder_model.optimizer.learning_rate, self.args.learning_rate_auto_encoder)

        history = self.autoencoder_model.fit(
            x_train, x_train, batch_size=self.args.batch_size,
            epochs=self.args.epochs_auto_encoder,
            validation_data=(x_val, x_val))
        self.history_dict = history.history

        return self.history_dict

    def save_models(self):
        self.encoder.save(self.save_path.joinpath("encoder"))
        self.decoder.save(self.save_path.joinpath("decoder"))
        self.autoencoder_model.save(self.save_path.joinpath("autoencoder"))
        np.save(self.save_path.joinpath(
            "autoencoder_history_dict.npy"), self.history_dict)

    def load_models(self):
        self.encoder = keras.models.load_model(
            self.save_path.joinpath("encoder"))
        self.decoder = keras.models.load_model(
            self.save_path.joinpath("decoder"))
        self.autoencoder_model = keras.models.load_model(
            self.save_path.joinpath("autoencoder"))
        self.history_dict = np.load(self.save_path.joinpath(
            "autoencoder_history_dict.npy"), allow_pickle=True).item()

    def evaluate(self, x_test, verbose=1):
        return self.autoencoder_model.evaluate(x_test, x_test, verbose=verbose)
