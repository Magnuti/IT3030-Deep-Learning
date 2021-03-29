from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pathlib

from config_parser import Arguments
from constants import Dataset
from autoencoder import AutoEncoder
from classifier import Classifier

if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()

    save_path = pathlib.Path("saves")
    save_path.mkdir(exist_ok=True)

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

    # Split data into labelled and unlabelled
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

    # TODO load/save depening on if the folder exists
    autoencoder = AutoEncoder(arguments, save_path)

    latent_vectors_before_training = autoencoder.encoder(x_test).numpy()

    # autoencoder_history_dict = autoencoder.train(
    #     x_train_unlabelled, x_test_unlabelled)
    # autoencoder.save_models()
    autoencoder.load_models()
    autoencoder_history_dict = autoencoder.history_dict

    latent_vectors_after_autoencoder_training = autoencoder.encoder(
        x_test).numpy()

    classifer = Classifier(arguments, num_classes,
                           autoencoder.encoder, save_path)
    # classifier_history_dict = classifer.train(
    #     x_train_labelled, y_train_labelled)
    # classifer.save_models()
    classifer.load_models()
    classifier_history_dict = classifer.history_dict

    latent_vectors_after_classifier_training = classifer.encoder(
        x_test).numpy()

    def plot_loss_and_accuracy():
        # TODO separate loss and accuracy according to spec
        # Plot accuracy and loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Model accuracy')
        plt.plot(autoencoder_history_dict['accuracy'])
        plt.plot(autoencoder_history_dict['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validaton'])

        plt.subplot(1, 2, 2)
        plt.title('Loss')
        plt.plot(autoencoder_history_dict['loss'])
        plt.plot(autoencoder_history_dict['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        plt.show()

    def plot_autoencoder_reconstructions(x_test):
        # Display images (from https://www.tensorflow.org/tutorials/generative/autoencoder)
        latent_vectors = autoencoder.encoder(x_test).numpy()
        decoded_imgs = autoencoder.decoder(latent_vectors).numpy()

        decoded_imgs = np.squeeze(decoded_imgs)  # Removes last 1 dimension
        x_test = np.squeeze(x_test)  # Removes last 1 dimension

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i])
            plt.title("Original")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i])
            plt.title("Reconstructed")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def get_cmap(n, name='hsv'):
        # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n + 1)

    def plot_latent_vector_clusters(n_vectors, latent_vectors_before_training,
                                    latent_vectors_after_autoencoder_training,
                                    latent_vectors_after_classifier_training,
                                    y_test):
        assert latent_vectors_before_training.shape == latent_vectors_after_autoencoder_training.shape and latent_vectors_before_training.shape == latent_vectors_after_classifier_training.shape

        indices = np.random.choice(
            latent_vectors_before_training.shape[0], n_vectors, replace=False)

        X_inputs_before_training = latent_vectors_before_training[indices]
        X_inputs_after_autoencoder_training = latent_vectors_after_autoencoder_training[
            indices]
        X_inputs_after_classifier_training = latent_vectors_after_classifier_training[
            indices]
        classes = y_test[indices]

        # Transforms X_inputs of shape (n_vectors, latent_vector_size) into (n_vectors, 2)
        X_embedded_before_training = TSNE(
            n_components=2).fit_transform(X_inputs_before_training)
        X_embedded_autoencoder_training = TSNE(n_components=2).fit_transform(
            X_inputs_after_autoencoder_training)
        X_embedded_classifier_training = TSNE(n_components=2).fit_transform(
            X_inputs_after_classifier_training)

        # One color for each class
        cmap = get_cmap(classes.shape[1])
        colors = []
        for i in range(classes.shape[0]):
            num = np.argmax(classes[i])
            colors.append(cmap(num))

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 3, 1)
        plt.title("Before training")
        plt.scatter(
            X_embedded_before_training[:, 0], X_embedded_before_training[:, 1], c=colors)
        plt.subplot(1, 3, 2)
        plt.title("After autoencoder training")
        plt.scatter(
            X_embedded_autoencoder_training[:, 0], X_embedded_autoencoder_training[:, 1], c=colors)
        plt.subplot(1, 3, 3)
        plt.title("After classifier training")
        plt.scatter(
            X_embedded_classifier_training[:, 0], X_embedded_classifier_training[:, 1], c=colors)
        plt.show()

    plot_autoencoder_reconstructions(x_test)
    plot_latent_vector_clusters(250, latent_vectors_before_training,
                                latent_vectors_after_autoencoder_training, latent_vectors_after_classifier_training, y_test)
