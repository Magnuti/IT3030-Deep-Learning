import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import pathlib
import numpy as np

image_path = pathlib.Path("images")
image_path.mkdir(exist_ok=True)


def plot_loss_and_accuracy(autoencoder_history_dict, classifier_history_dict, supervised_classifier_history_dict):
    # Plot autoencoder loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Autoencoder loss')
    plt.plot(autoencoder_history_dict['loss'])
    plt.plot(autoencoder_history_dict['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validaton'])

    # Plot classifier accuracies
    plt.subplot(1, 2, 2)
    plt.title('Comparative Classifier Accuracy')
    plt.plot(classifier_history_dict['accuracy'])
    plt.plot(classifier_history_dict['val_accuracy'])
    plt.plot(
        supervised_classifier_history_dict['accuracy'])
    plt.plot(
        supervised_classifier_history_dict['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Semi training accuracy', 'Semi val. accuracy',
                "Supervised training accuracy", "Supervised val. accuracy"])

    plt.savefig(image_path.joinpath("loss_and_accuracy.png"))
    plt.show()


def plot_autoencoder_reconstructions(x_test, reconstructed_images):
    # Display images (from https://www.tensorflow.org/tutorials/generative/autoencoder)

    assert len(x_test) == len(reconstructed_images)

    plt.figure(figsize=(20, 4))
    n = len(x_test)
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
        plt.imshow(reconstructed_images[i])
        plt.title("Reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(image_path.joinpath("autoencoder_reconstructions.png"))
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
    assert latent_vectors_before_training.shape == latent_vectors_after_autoencoder_training.shape
    assert latent_vectors_before_training.shape == latent_vectors_after_classifier_training.shape
    assert latent_vectors_before_training.shape[0] == y_test.shape[0]

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

    plt.legend(handles=[mpatches.Patch(color=cmap(i), label=i)
                        for i in range(10)], bbox_to_anchor=(1.25, 1), loc='upper right')
    plt.savefig(image_path.joinpath("latent_vector_clusters.png"))
    plt.show()
