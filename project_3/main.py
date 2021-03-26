from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from config_parser import Arguments
from constants import Dataset
from autoencoder import AutoEncoder

if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()

    if arguments.dataset == Dataset.MNIST:
        # From https://keras.io/examples/vision/mnist_convnet/

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        num_classes = 10
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    elif arguments.dataset == Dataset.FASHION_MNIST:
        # From https://keras.io/examples/vision/mnist_convnet/

        # the data, split between train and test sets
        (x_train, y_train), (x_test,
                             y_test) = keras.datasets.fashion_mnist.load_data()

        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        num_classes = 10
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
    else:
        raise NotImplementedError()

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    train_split_index = int(x_train.shape[0] * arguments.split_ratio)
    x_train_labelled = x_train[:train_split_index]
    x_train_unlabelled = x_train[train_split_index:]

    y_train_labelled = y_train[:train_split_index]
    y_train_unlabelled = y_train[train_split_index:]

    test_split_index = int(x_test.shape[0] * arguments.split_ratio)
    x_test_labelled = x_test[:test_split_index]
    x_test_unlabelled = x_test[test_split_index:]

    y_test_labelled = y_train[:test_split_index]
    y_test_unlabelled = y_train[test_split_index:]

    # print(x_train_labelled.shape)
    # print(x_train_unlabelled.shape)
    # print(y_train_labelled.shape)
    # print(y_train_unlabelled.shape)

    # print(x_test_labelled.shape)
    # print(x_test_unlabelled.shape)
    # print(y_test_labelled.shape)
    # print(y_test_unlabelled.shape)

    autoencoder = AutoEncoder(arguments)
    # autoencoder.train(x_train_unlabelled)
    # autoencoder.save_models()
    autoencoder.load_models()

    # Display images
    encoded_imgs = autoencoder.encoder(x_test).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    decoded_imgs = np.squeeze(decoded_imgs)  # Removes last 1 dimension
    x_test = np.squeeze(x_test)  # Removes last 1 dimension

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
