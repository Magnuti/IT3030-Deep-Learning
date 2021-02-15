import numpy as np
import matplotlib.pyplot as plt


def split_dataset(X, Y, training_ratio=0.7, validation_ratio=0.2, test_ratio=0.1):
    """
    Splits the dataset into training, validation and testing data.

    Args
        X: inputs of shape (dataset_size, input_size)
        Y: targets of shape (dataset_size, output_size)
        training_ratio, validation_ratio and test_ratio must sum to 1
    Returns
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """

    dataset_size = X.shape[0]
    assert dataset_size == Y.shape[0]
    assert round(training_ratio + validation_ratio + test_ratio,
                 4) == 1.0, "Actual value {}".format(round(training_ratio + validation_ratio + test_ratio, 4))

    training_index = int(np.floor(dataset_size * training_ratio))
    validation_index = int(
        np.floor(dataset_size * validation_ratio)) + training_index
    testing_index = int(
        np.floor(dataset_size * test_ratio)) + validation_index

    return (X[0:training_index], Y[0:training_index],
            X[training_index:validation_index], Y[training_index:validation_index],
            X[validation_index:testing_index], Y[validation_index:testing_index])


def shuffle_data_and_targets(X, Y):
    """
    Shuffles data and target arrays in the same order along axis 0.

    Args
        X: np.ndarray
        Y: np.ndarray with of the same shape as X

    Returns
        X, Y: np.ndarray shuffled
    """
    assert X.shape[0] == Y.shape[0], "Shapes are not equal along axis 0: {}, {}".format(
        X.shape, Y.shape)

    shuffle_indexes = np.arange(X.shape[0])
    np.random.shuffle(shuffle_indexes)
    X = X[shuffle_indexes]
    Y = Y[shuffle_indexes]
    return X, Y


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
