from tensorflow import keras
import numpy as np
import pathlib

from config_parser import Arguments
from constants import Dataset
from autoencoder import AutoEncoder
from classifier import Classifier
from supervised_classifier import SupervisedClassifier
from plotting import plot_loss_and_accuracy, plot_latent_vector_clusters, plot_autoencoder_reconstructions

if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()

    save_path = pathlib.Path("saves")
    save_path.mkdir(exist_ok=True)

    def load_data(dataset):
        if dataset == Dataset.MNIST:
            return keras.datasets.mnist.load_data()
        elif dataset == Dataset.FASHION_MNIST:
            return keras.datasets.fashion_mnist.load_data()
        elif dataset == Dataset.CIFAR10:
            return keras.datasets.cifar10.load_data()
        elif dataset == Dataset.CIFAR100:
            return keras.datasets.cifar100.load_data()
        else:
            raise NotImplementedError()

    def pre_process_image_data(x):
        # Scale images to the [0, 1] range
        return x.astype("float32") / 255

    def split_dataset(x, y, split_ratio):
        assert x.shape[0] == y.shape[0]

        split_index = int(x.shape[0] * split_ratio)
        x_0 = x[:split_index]
        y_0 = y[:split_index]
        x_1 = x[split_index:]
        y_1 = y[split_index:]

        return (x_0, y_0), (x_1, y_1)

    def split_dataset_x_only(x, split_ratio):
        split_index = int(x.shape[0] * split_ratio)
        return x[:split_index], x[split_index:]

    (x_train, y_train), (x_test, y_test) = load_data(arguments.dataset)

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    if arguments.dataset == Dataset.MNIST or arguments.dataset == Dataset.FASHION_MNIST:
        # Make sure images go from shape (28, 28) to (28, 28, 1)
        x = np.expand_dims(x, -1)

    if arguments.dataset == Dataset.MNIST:
        num_classes = 10
    elif arguments.dataset == Dataset.FASHION_MNIST:
        num_classes = 10
    elif arguments.dataset == Dataset.CIFAR10:
        num_classes = 10
    elif arguments.dataset == Dataset.CIFAR100:
        num_classes = 100
    else:
        raise NotImplementedError()

    x = pre_process_image_data(x)

    # One hot encode targets
    y = keras.utils.to_categorical(y, num_classes)

    input_shape = x.shape[1:]

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("Input shape:", input_shape)

    (x_labeled, y_labeled), (x_unlabeled, _) = split_dataset(
        x, y, arguments.labeled_to_unlabeled_split_ratio)

    print("Labeled:", x_labeled.shape[0],
          "to unlabeled:", x_unlabeled.shape[0])

    (x_train_labeled, y_train_labeled), (x_test_labeled, y_test_labeled) = split_dataset(
        x_labeled, y_labeled, arguments.train_to_test_ratio)
    (x_train_labeled, y_train_labeled), (x_val_labeled, y_val_labeled) = split_dataset(
        x_train_labeled, y_train_labeled, 1.0 - arguments.validation_ratio)

    (x_train_unlabeled, x_test_unlabeled) = split_dataset_x_only(
        x_unlabeled, arguments.train_to_test_ratio)
    (x_train_unlabeled, x_val_unlabeled) = split_dataset_x_only(
        x_train_unlabeled, arguments.train_to_test_ratio)

    print("Labeled data:")
    print("\tTraining data:", x_train_labeled.shape, y_train_labeled.shape)
    print("\tValidation data:", x_val_labeled.shape, y_val_labeled.shape)
    print("\tTesting data:", x_test_labeled.shape, y_test_labeled.shape)

    print("Unlabeled data:")
    print("\tTraining data:", x_train_unlabeled.shape)
    print("\tValidation data:", x_val_unlabeled.shape)
    print("\tTesting data:", x_test_unlabeled.shape)

    autoencoder = AutoEncoder(arguments, save_path, input_shape)

    latent_vectors_before_training = autoencoder.encoder(
        x_test_labeled).numpy()

    # May fix these later, good for now
    load_autoencoder = False
    load_classifier = False
    load_supervised_classifer = False

    if not load_autoencoder:
        autoencoder_history_dict = autoencoder.train(
            x_train_unlabeled, x_val_unlabeled)
        autoencoder.save_models()
    else:
        autoencoder.load_models()
        autoencoder_history_dict = autoencoder.history_dict

    latent_vectors_after_autoencoder_training = autoencoder.encoder(
        x_test_labeled).numpy()

    classifer = Classifier(arguments, num_classes,
                           autoencoder.encoder, save_path)
    if not load_classifier:
        classifier_history_dict = classifer.train(
            x_train_labeled, y_train_labeled, x_val_labeled, y_val_labeled)
        classifer.save_models()
    else:
        classifer.load_models()
        classifier_history_dict = classifer.history_dict

    supervised_classifier = SupervisedClassifier(
        arguments, num_classes, save_path, input_shape)

    if not load_supervised_classifer:
        supervised_classifier_history_dict = supervised_classifier.train(
            x_train_labeled, y_train_labeled, x_val_labeled, y_val_labeled)
        supervised_classifier.save_models()
    else:
        supervised_classifier.load_models()
        supervised_classifier_history_dict = supervised_classifier.history_dict

    latent_vectors_after_classifier_training = classifer.encoder(
        x_test_labeled).numpy()

    # Evaluation on test set
    autoencoder_score = autoencoder.evaluate(x_test_unlabeled)
    classifier_score = classifer.evaluate(x_test_labeled, y_test_labeled)
    supervised_classifier_score = supervised_classifier.evaluate(
        x_test_labeled, y_test_labeled)
    print("Autoencoder test loss:", autoencoder_score)
    print("Semi-supervised classifier test [loss, accuracy]", classifier_score)
    print(
        "Supervised classifier test [loss, accuracy]", supervised_classifier_score)

    if arguments.visualize:
        plot_loss_and_accuracy(autoencoder_history_dict,
                               classifier_history_dict, supervised_classifier_history_dict)

        latent_vectors = autoencoder.encoder(x_test_labeled).numpy()
        reconstructed_images = autoencoder.decoder(latent_vectors).numpy()

        # Removes last 1 dimension
        reconstructed_images = np.squeeze(reconstructed_images)
        x_test_labeled = np.squeeze(x_test_labeled)

        plot_autoencoder_reconstructions(
            x_test_labeled, reconstructed_images)

        plot_latent_vector_clusters(250, latent_vectors_before_training,
                                    latent_vectors_after_autoencoder_training,
                                    latent_vectors_after_classifier_training,
                                    y_test_labeled)
