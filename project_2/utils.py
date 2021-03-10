import numpy as np
import matplotlib.pyplot as plt


def split_dataset(XY, training_ratio=0.7, validation_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset into training, validation and testing data.

    Args
        XY: list of sequence cases, each with a tuple of input-output pairs
        training_ratio, validation_ratio and test_ratio must sum to 1
    Returns
        X_train, X_val, X_test
    """

    dataset_size = len(XY)
    assert round(training_ratio + validation_ratio + test_ratio,
                 4) == 1.0, "Actual value {}".format(round(training_ratio + validation_ratio + test_ratio, 4))

    training_index = int(np.floor(dataset_size * training_ratio))
    validation_index = int(
        np.floor(dataset_size * validation_ratio)) + training_index
    testing_index = int(
        np.floor(dataset_size * test_ratio)) + validation_index

    return (XY[0:training_index], XY[training_index:validation_index], XY[validation_index:testing_index])


def one_hot_encode(X):
    """
    One-hot encodes the input

    Args
        X: np.ndarray of shape (number of outputs, batch_size)

    Return
        one-hot encoded version of X
    """
    output = np.zeros_like(X)
    max_indexes = np.argmax(X, axis=0)
    output[max_indexes, np.arange(X.shape[1])] = 1
    return output


def plot_loss(train_loss_history, val_loss_history):
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(train_loss_history)), train_loss_history,
             label="Training loss")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(val_loss_history)),
             val_loss_history, label="Validation loss")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Loss")

    plt.show()


def plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history):
    # plt.ylim([0.2, .6])
    # utils.plot_loss(train_history["loss"],
    #                 "Training Loss", npoints_to_average=10)
    # utils.plot_loss(val_history["loss"], "Validation Loss")

    plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(len(train_loss_history)), train_loss_history,
             label="Training loss")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(len(val_loss_history)),
             val_loss_history, label="Validation loss")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 3)
    plt.plot(np.arange(len(train_accuracy_history)),
             train_accuracy_history, label="Training accuracy")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Accuracy")

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(len(val_accuracy_history)),
             val_accuracy_history, label="Validation accuracy")
    plt.legend()
    plt.xlabel("Iterations (number of batches)")
    plt.ylabel("Accuracy")

    plt.show()
