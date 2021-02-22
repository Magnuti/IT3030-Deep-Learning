import numpy as np

from constants import ActivationFunction, LossFunction
from layers import Layer
from utils import split_dataset, one_hot_encode, plot_loss_and_accuracy


def cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Args:
        outputs: outputs of model of shape: (num_classes, batch_size)
        targets: labels/targets of each image of shape: (num_classes, batch_size)
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = - np.sum(targets * np.log(outputs), axis=0)
    return np.average(loss)


def cross_entropy_loss_derivative(outputs: np.ndarray, targets: np.ndarray):
    """
    Args:
        outputs: outputs of model of shape: (num_classes, batch_size)
        targets: same shape as outputs
    Returns:
        derivatives/gradients of the same shape as outputs
    """
    # Taken from the course instructors post
    return np.where(outputs != 0, -targets / outputs, 0.0)


def MSE(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Args:
        outputs: outputs of model of shape: (number_of_outputs, batch_size)
        targets: labels/targets of each image of shape: (number_of_outputs, batch size)
    Returns:
        Mean squared error (float)
    """
    assert outputs.shape == targets.shape

    return np.average(np.average((outputs - targets)**2, axis=0))


def MSE_derivative(outputs, targets):
    """
    Args
        outputs: np.ndarray of shape (number_of_outputs, batch_size)
        targets: np.ndarray of shape (number_of_outputs, batch size)
    Returns
        np.ndarray of shape (number_of_outputs, batch_size)
    """

    number_of_outputs = outputs.shape[0]
    # ? negative or positive return? Maybe add a minus sign before retur
    return 2.0 / number_of_outputs * (outputs - targets)


def run_loss_function(loss_function, outputs, targets):
    if(loss_function == LossFunction.MSE):
        return MSE(outputs, targets)
    elif(loss_function == LossFunction.CROSS_ENTROPY):
        return cross_entropy_loss(outputs, targets)
    else:
        raise NotImplementedError()


def derivative_loss_function(loss_function, outputs, targets):
    if(loss_function == LossFunction.MSE):
        return MSE_derivative(outputs, targets)
    elif(loss_function == LossFunction.CROSS_ENTROPY):
        return cross_entropy_loss_derivative(outputs, targets)
    else:
        raise NotImplementedError()


def accuracy(outputs, targets):
    """
    Calculates the accuracy (i.e., correct predictions / total predictions).

    Args:
        outputs: np.ndarray of shape (number_of_outputs, batch_size)
        targets: same dimension as outputs
    Return
        accuracy: float
    """
    assert outputs.shape == targets.shape, "Shape was {}, {}".format(
        outputs.shape, targets.shape)

    outputs = one_hot_encode(outputs)
    dataset_size = outputs.shape[1]
    correct_predictions = 0
    for i in range(dataset_size):
        if np.all(np.equal(outputs[:, i], targets[:, i])):
            correct_predictions += 1

    return correct_predictions / dataset_size


class NeuralNetwork:
    def __init__(self, learning_rate, neurons_in_each_layer, activation_functions, softmax,
                 loss_function, global_weight_regularization_option, global_weight_regularization_rate,
                 initial_weight_ranges, initial_bias_ranges, verbose):
        self.learning_rate = learning_rate

        if softmax:
            # Appends a SoftMax layer as the last layer with as many neurons as the last layer before SoftMax
            neurons_in_each_layer.append(neurons_in_each_layer[-1])

        self.neurons_in_each_layer = neurons_in_each_layer
        # Add None as activation function to the input layer and SoftMax to last layer
        activation_functions.insert(0, None)
        if softmax:
            activation_functions.append(ActivationFunction.SOFTMAX)
        self.activation_functions = activation_functions
        self.softmax = softmax
        self.loss_function = loss_function
        self.global_weight_regularization_option = global_weight_regularization_option
        self.global_weight_regularization_rate = global_weight_regularization_rate
        self.initial_weight_ranges = initial_weight_ranges
        self.initial_bias_ranges = initial_bias_ranges
        self.verbose = verbose

        self.layers = self.__build_network()

        if(self.verbose):
            for i, layer in enumerate(self.layers):
                if i == 0:
                    print("Input layer")
                    continue

                print("Layer {}: {}".format(i, layer))

    def __build_network(self):
        layers = [None]
        neurons_in_previous_layer = self.neurons_in_each_layer[0]
        for i, neuron_count in enumerate(self.neurons_in_each_layer):
            if i == 0:
                continue  # Skip input layer

            layers.append(Layer(neuron_count, neurons_in_previous_layer,
                                self.activation_functions[i], self.learning_rate, self.initial_weight_ranges,
                                self.initial_bias_ranges, verbose=self.verbose, name="Dense {}".format(i)))
            neurons_in_previous_layer = neuron_count

        return layers

    def batch_loader(self, X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle=True):
        """
        Creates a batch generator over the whole dataset (X, Y) which returns a generator iterating over all the batches.
        This function is called once each epoch.

        Args:
            X: inputs of shape (dataset_size, input_size)
            Y: labels of shape (dataset_size, output_size)
            shuffle (bool): To shuffle the dataset between each epoch or not.
        """
        assert X.shape[0] == Y.shape[0], "Inputs and targets must be of same length. X: {}, Y:{}".format(
            X.shape, Y.shape)

        dataset_size = X.shape[0]
        if dataset_size % batch_size == 0:
            num_batches = dataset_size // batch_size
        else:
            # Drop last batch if the dataset is not evenly divisible by the batch size
            num_batches = int(np.ceil(dataset_size / batch_size))

        indices = list(range(dataset_size))

        if(shuffle):
            np.random.shuffle(indices)

        for i in range(num_batches):
            # Divides the indicies into into batches
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            x = X[batch_indices]
            y = Y[batch_indices]
            yield (x, y)

    def forward_pass(self, X: np.ndarray):
        """
        Performs forward pass over a mini-batch.

        Args
            X: np.ndarray of shape (number_of_input_nodes, batch_size)
        Returns
            np.ndarray of shape (number_of_output_nodes, batch_size)
        """

        outputs = X  # Initial inputs
        # output_history = []
        for i, layer in enumerate(self.layers):
            if(i == 0):
                continue

            outputs = layer.forward_pass(outputs)
            # output_history.append(outputs)

        return outputs

    def backward_pass(self, outputs, targets):
        """
        Performs backward pass over a mini-batch.

        Args
            outputs: np.ndarray of shape (output_size, batch_size)
            targets: np.ndarray the same shape as outputs
        """
        assert outputs.shape == targets.shape

        # Jacobian loss
        R = derivative_loss_function(self.loss_function, outputs, targets).T
        for i, layer in reversed(list(enumerate(self.layers))):
            # Skip input layer
            if layer is not None:
                R = layer.backward_pass(R)

    def train(self, epochs, batch_size, X_train, Y_train, X_val, Y_val, shuffle=True):
        """
        Performs the training phase (forward pass + backward propagation) over a number of epochs.

        Args
            epochs: int
            X_train: np.ndarray of shape (training dataset size, input_size)
            Y_train: np.ndarray of shape (training dataset size, output_size)
            X_val: np.ndarray of shape (validation dataset size, input_size)
            Y_val: np.ndarray of shape (validation dataset size, output_size)
            shuffle: bool, whether or not to shuffle the dataset
        """
        print("Training over {} epochs with a batch size of {}".format(
            epochs, batch_size))

        # Transpose X and Y because we want them as column vectors
        X_val = X_val.T
        Y_val = Y_val.T

        train_loss_history = []
        train_accuracy_history = []
        val_loss_history = []
        val_accuracy_history = []

        iteration = 0
        # Validate every time we run through 20 % of the datset
        # iterations_per_validation = (
        #     X_train.shape[0] // batch_size) // 5  # todo fix %
        for epoch in range(epochs):
            train_loader = self.batch_loader(
                X_train, Y_train, batch_size, shuffle=shuffle)
            for X_batch, Y_batch in iter(train_loader):
                # Transpose X and Y because we want them as column vectors
                X_batch = X_batch.T
                Y_batch = Y_batch.T
                output_train = self.forward_pass(X_batch)
                self.backward_pass(output_train, Y_batch)
                loss_train = run_loss_function(
                    self.loss_function, output_train, Y_batch)

                train_loss_history.append(loss_train)

                # Validation step at every iteration
                # if iteration % iterations_per_validation == 0:
                output_val = self.forward_pass(X_val)

                loss_val = run_loss_function(
                    self.loss_function, output_val, Y_val)
                accuracy_train = accuracy(output_train, Y_batch)
                accuracy_val = accuracy(output_val, Y_val)

                val_loss_history.append(loss_val)
                train_accuracy_history.append(accuracy_train)
                val_accuracy_history.append(accuracy_val)

                print("Epoch: {}, iteration: {}, training loss: {}, validation loss {}:".format(
                    epoch, iteration, loss_train, loss_val))

                iteration += 1
                # if iteration >= 2:  # !
                #     exit()

        return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

    def final_accuracy(self, X_test, Y_test):
        """
        Calculates the prediction for a test set

        Args
            X_test: np.ndarray of shape (batch_size, input_size)
            Y_test: np.ndarray of shape (batch_size, output_size)

        Return
            accuracy: float
        """
        output = self.forward_pass(X_test.T)
        return accuracy(output, Y_test.T)

    def predict(self, X, one_hot=True):
        """
        Feeds X into the model and returns the output.

        Args
            X: np.ndarray of shape (batch_size, input_size)
            one_hot: bool, whether the output should be one-hot encoded.

        Return
            np.ndarray of shape (batch_size, output_size)
        """
        output = self.forward_pass(X.T)
        if one_hot:
            output = one_hot_encode(output)
        return output.T


if __name__ == "__main__":
    learning_rate = 0.1
    neurons_in_each_layer = [2, 3, 3, 2]
    activation_functions = [
        ActivationFunction.RELU, ActivationFunction.RELU, ActivationFunction.LINEAR]
    softmax = True
    loss_function = LossFunction.CROSS_ENTROPY
    global_weight_regularization_option = None
    global_weight_regularization_rate = None
    initial_weight_ranges = "glorot_normal"
    initial_bias_ranges = [0, 0]
    verbose = False
    nn = NeuralNetwork(learning_rate, neurons_in_each_layer, activation_functions, softmax, loss_function,
                       global_weight_regularization_option, global_weight_regularization_rate, initial_weight_ranges,
                       initial_bias_ranges, verbose)

    def XOR_data(count):
        X = np.empty((count, 2))
        targets = np.empty((count, 2))

        index = 0
        # 01 = 0
        # 10 = 1 # lmao
        for i in range(count // 4):
            for j in range(2):
                # X[index, j] = np.random.uniform(0.0, 0.5)
                X[index, j] = 0.0

            targets[index, 0] = 0.0
            targets[index, 1] = 1.0
            index += 1

        # (1, 1) = 0
        for i in range(count // 4):
            for j in range(2):
                # X[index, j] = np.random.uniform(0.5, 1.0)
                X[index, j] = 1.0

            targets[index, 0] = 0.0
            targets[index, 1] = 1.0
            index += 1

        # (0, 1) = 1
        for i in range(count // 4):
            # X[index, 0] = np.random.uniform(0.0, 0.5)
            # X[index, 1] = np.random.uniform(0.5, 1.0)
            X[index, 0] = 0.0
            X[index, 1] = 1.0
            targets[index, 0] = 1.0
            targets[index, 1] = 0.0
            index += 1

        # (1, 0) = 1
        for i in range(count // 4):
            # X[index, 0] = np.random.uniform(0.5, 1.0)
            # X[index, 1] = np.random.uniform(0.0, 0.5)
            X[index, 0] = 1.0
            X[index, 1] = 0.0
            targets[index, 0] = 1.0
            targets[index, 1] = 0.0
            index += 1

        shuffle_indexes = np.arange(count)
        np.random.shuffle(shuffle_indexes)
        X = X[shuffle_indexes]
        targets = targets[shuffle_indexes]

        return X, targets

    epochs = 40
    batch_size = 64
    X, Y = XOR_data(20000)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    # print("Outputs:\n", X)

    # print("Targets:\n", targets)

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = nn.train(
        epochs, batch_size, X_train, Y_train, X_val, Y_val)

    plot_loss_and_accuracy(
        train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history)

    input_check = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    prediction = nn.predict(input_check)
    prediction = one_hot_encode(prediction)
    target_check = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    print("Actual prediction\n", prediction)
    print("Desired prediction\n", target_check)
