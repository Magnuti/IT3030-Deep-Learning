import numpy as np
import matplotlib.pyplot as plt

from constants import ActivationFunction, LossFunction

# TODO make these elif functions into a class maybe


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.power(np.tanh(x), 2)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones(x.shape)


def softmax_fun(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.array of shape(num_classes, batch_size)

    Returns:
        np.array that sum to 1 of the same shape as x
    """

    # We want to avoid computing e^x when x is very large, as this scales exponentially
    # Therefore, we can use a trick to "lower" all values of x, but still be able to
    # achieve the same result as discussed here
    # https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
    # and here https://stackoverflow.com/questions/43401593/softmax-of-a-large-number-errors-out
    # We do this by subtracting the max values for each batch.
    x -= np.amax(x, axis=0)
    x = np.exp(x)
    return x / x.sum(axis=0)


def softmax_derivative(inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


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

# def binary_cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
#     """
#     Args:
#         targets: labels/targets of each image of shape: [batch size, 1]
#         outputs: outputs of model of shape: [batch size, 1]
#     Returns:
#         Cross entropy error (float)
#     """
#     assert targets.shape == outputs.shape,\
#         f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

#     batch_size = targets.shape[0]
#     loss = np.zeros(batch_size)
#     for i in range(batch_size):
#         y = targets[i, 0]
#         y_hat = outputs[i, 0]
#         loss[i] = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

#     return np.average(loss)


def binary_cross_entropy_loss(outputs: np.ndarray, targets: np.ndarray) -> float:
    """
    Args:
        outputs: outputs of model of shape: (batch size)
        targets: labels/targets of each image of shape: (batch size)
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape, "Targets shape: {}, outputs: {}".format(
        targets.shape, outputs.shape)

    loss = - (targets * np.log(outputs)) + (1 - targets) * np.log(1 - outputs)
    return np.average(loss)

    # batch_size = targets.shape
    # loss = np.zeros(batch_size)

    # for i in range(batch_size):
    #     target = targets[i]
    #     output = outputs[i]
    #     loss[i] = -(target * np.log(output) +
    #                 (1 - target) * np.log(1 - output))

    # return np.average(loss)


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


def run_activation_function(af, inputs):
    if(af == ActivationFunction.SIGMOID):
        return sigmoid(inputs)
    elif(af == ActivationFunction.TANH):
        return tanh(inputs)
    elif(af == ActivationFunction.RELU):
        return relu(inputs)
    elif(af == ActivationFunction.LINEAR):
        return linear(inputs)
    elif(af == ActivationFunction.SOFTMAX):
        return softmax_fun(inputs)
    else:
        raise NotImplementedError()


def derivative_activation_function(af, x):
    if(af == ActivationFunction.SIGMOID):
        return sigmoid_derivative(x)
    elif(af == ActivationFunction.TANH):
        return tanh_derivative(x)
    elif(af == ActivationFunction.RELU):
        return relu_derivative(x)
    elif(af == ActivationFunction.LINEAR):
        return linear_derivative(x)
    elif(af == ActivationFunction.SOFTMAX):
        return softmax_derivative(x)
    else:
        raise NotImplementedError()


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


def glorot_normal(neurons_in_previous_layer, neurons_in_this_layer):
    standard_deviation = np.sqrt(
        2 / (neurons_in_previous_layer + neurons_in_this_layer))
    return np.random.normal(0.0, standard_deviation,
                            size=(neurons_in_previous_layer, neurons_in_this_layer))


def glorot_uniform(neurons_in_previous_layer, neurons_in_this_layer):
    standard_deviation = np.sqrt(
        6.0 / (neurons_in_previous_layer + neurons_in_this_layer))
    return np.random.uniform(-standard_deviation, standard_deviation,
                             size=(neurons_in_previous_layer, neurons_in_this_layer))


def init_weights_with_range(low, high, neurons_in_previous_layer, neurons_in_this_layer):
    return np.random.uniform(low, high, size=(neurons_in_previous_layer, neurons_in_this_layer))


class Layer:

    def __init__(self, neuron_count, neurons_in_previous_layer, activation_function, learning_rate,
                 initial_weight_ranges, initial_bias_ranges, verbose=False, name=""):

        self.neuron_count = neuron_count

        if initial_weight_ranges == "glorot_normal":
            self.weights = glorot_normal(
                neurons_in_previous_layer, neuron_count)
        elif initial_weight_ranges == "glorot_uniform":
            self.weights = glorot_uniform(
                neurons_in_previous_layer, neuron_count)
        else:
            self.weights = init_weights_with_range(
                initial_weight_ranges[0], initial_weight_ranges[1], neurons_in_previous_layer, neuron_count)

        # One bias per neuron, column vector
        self.bias = np.random.uniform(
            initial_bias_ranges[0], initial_bias_ranges[1], size=(neuron_count, 1))
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.name = name

    def forward_pass(self, X):
        """
        Args:
            X: np.ndarray of shape (self.neurons_in_previous_layer, batch_size)
        Returns:
            np.ndarray of shape (self.neuron_count, batch_size)
        """
        weighted_inputs = np.matmul(self.weights.T, X) + self.bias
        self.prev_layer_outputs = X
        # self.weighted_inputs = weighted_inputs
        self.activations = run_activation_function(
            self.activation_function, weighted_inputs)

        if self.verbose:
            print("Forward passing layer", self.name)
            print("Input (input_size, batch_size):\n", X)
            print("Weights:\n", self.weights)
            print("Bias:\n", self.bias)
            print("Output:\n", self.activations)

        return self.activations

    def backward_pass(self, R):
        # TODO transpose all R to end this mess
        # print("R", R.shape)
        if self.activation_function is not None:
            if self.activation_function == ActivationFunction.SOFTMAX:
                # # Hardcoded J^L_S for now with SoftMax +  MSE
                # # Row vector of shape (batch_size, number_of_outputs)

                # # j_soft == JSZ
                # # Builds the J^Soft matrix
                # j_soft = np.empty((self.neuron_count, self.neuron_count))
                # for i in range(self.neuron_count):
                #     for j in range(self.neuron_count):
                #         if i == j:
                #             j_soft[i, j] = self.activations[i] - \
                #                 self.activations[i] ** 2
                #         else:
                #             j_soft[i, j] = - self.activations[i] * \
                #                 self.activations[j]

                # # Transpose R because we need it as a column vector
                # R = np.dot(R.T, j_soft)
                # TODO fix the ripoff, note here we transpose both R and activations
                # self.activations has shape (batch_size, activations)

                activations_T = self.activations.T
                temp_gradient_T = R.T
                act_shape = activations_T.shape
                act = activations_T.reshape(act_shape[0], 1, act_shape[-1])

                jacobian = - (act.transpose((0, 2, 1)) @ act) * \
                    (1 - np.identity(activations_T.shape[-1]))
                jacobian += np.identity(act_shape[-1]) * \
                    (act * (1 - act)).transpose((0, 2, 1))

                gradient = (
                    jacobian @ temp_gradient_T.reshape(act_shape[0], act_shape[-1], 1))
                gradient = gradient.reshape((act_shape[0], act_shape[-1]))

                R = gradient
                # print("R", R)
            else:
                activation_gradient = derivative_activation_function(
                    self.activation_function, self.activations).T
                # print("AG", activation_gradient.shape)
                R = activation_gradient * R  # ? * here??

        # Gradients for weights and bias
        batch_size = R.shape[0]
        # print("batch_size", batch_size)
        # print("Prev layer output", self.prev_layer_outputs.shape)
        # Divide by batch_size to get the average gradients over the batch
        # The average works because matrix multiplication sums the gradients
        gradient_weights = (self.prev_layer_outputs @ R) / \
            batch_size  # ! not .T here
        gradient_bias = (R.T.sum(axis=-1, keepdims=True)) / batch_size

        self.weights -= self.learning_rate * gradient_weights
        self.bias -= self.learning_rate * gradient_bias

        # next_R = np.transpose(self.weights @ R.T)
        return np.matmul(R, self.weights.T)

    def __str__(self):
        return "{} neurons with {} as activation function".format(self.neuron_count, self.activation_function)


class NeuralNetwork:
    def __init__(self, learning_rate, neurons_in_each_layer, activation_functions,
                 loss_function, global_weight_regularization_option, global_weight_regularization_rate,
                 initial_weight_ranges, initial_bias_ranges, verbose):
        self.learning_rate = learning_rate
        self.neurons_in_each_layer = neurons_in_each_layer
        # Add None as af to the input layer
        activation_functions.insert(0, None)
        self.activation_functions = activation_functions
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

    def batch_loader(self, X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle=False):
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

    def get_new_batch(self, batch_size):
        # TODO get a random sample of the training data, see CV assignment 1
        raise NotImplementedError

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
            outputs: np.ndarray of shape (?)
            targets: np.ndarray of shape (?)
        Returns
            None
        """

        R = derivative_loss_function(self.loss_function, outputs, targets).T

        # print("Original R:\n", R)
        for i, layer in reversed(list(enumerate(self.layers))):
            # print("Layer", i, layer)
            if layer is not None:
                R = layer.backward_pass(R)
                # print(R)

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
        iterations_per_validation = (
            X_train.shape[0] // batch_size) // 5  # todo fix %
        for epoch in range(epochs):
            train_loader = self.batch_loader(
                X_train, Y_train, batch_size, shuffle=shuffle)
            for X_batch, Y_batch in iter(train_loader):
                # Transpose X and Y because we want them as column vectors
                X_batch = X_batch.T
                Y_batch = Y_batch.T
                output_train = self.forward_pass(X_batch)
                nn.backward_pass(output_train, Y_batch)
                loss_train = run_loss_function(
                    loss_function, output_train, Y_batch)

                train_loss_history.append(loss_train)

                # Validation step every TODO epoch
                if iteration % iterations_per_validation == 0:
                    # print("Validating...")
                    output_val = self.forward_pass(X_val)

                    loss_val = run_loss_function(
                        loss_function, output_val, Y_val)
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

    def predict(self, X):
        """
        Performs classification over X

        Args
            X: np.ndarray of shape (dataset size, input_size)
        """
        # TODO one-hot encode the output?
        return self.forward_pass(X.T)


if __name__ == "__main__":
    # X = np.array([[0.1, 0.2, 0.51, 0.3], [
    #              0.8, 0.7, 0.49, 0.1], [0.1, 0.1, 0.0, 0.6]])
    # print(X)
    # X = one_hot_encode(X)
    # print(X)
    # exit()
    # learning_rate = 0.1
    # neurons_in_each_layer = [2, 2, 2]
    # activation_functions = [
    #     ActivationFunction.SIGMOID, ActivationFunction.SOFTMAX]
    # loss_function = LossFunction.CROSS_ENTROPY
    # global_weight_regularization_option = None
    # global_weight_regularization_rate = None
    # initial_weight_ranges = None
    # initial_bias_ranges = None
    # softmax = False
    # verbose = True
    # nn = NeuralNetwork(learning_rate, neurons_in_each_layer, activation_functions, loss_function,
    #                    global_weight_regularization_option, global_weight_regularization_rate, initial_weight_ranges,
    #                    initial_bias_ranges, softmax, verbose)

    # X_train = np.array([[1, 1], [1, 1]])
    # Y_train = np.array([[1, 1], [0, 0]])
    # print(X_train)
    # print(Y_train)
    # for i in range(5):
    #     output = nn.forward_pass(X_train)
    #     print("Output:\n", output)
    #     loss = run_loss_function(loss_function, output, Y_train)
    #     print("Loss:", round(loss, 6))
    #     nn.backward_pass(output, Y_train)

    # exit()

    learning_rate = 0.1
    neurons_in_each_layer = [2, 3, 3, 2]
    activation_functions = [
        ActivationFunction.RELU, ActivationFunction.RELU, ActivationFunction.SOFTMAX]
    loss_function = LossFunction.MSE
    global_weight_regularization_option = None
    global_weight_regularization_rate = None
    initial_weight_ranges = "glorot_normal"
    initial_bias_ranges = [0, 0]
    verbose = False
    nn = NeuralNetwork(learning_rate, neurons_in_each_layer, activation_functions, loss_function,
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
    epochs = 200
    batch_size = 64
    X, Y = XOR_data(20000)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)

    # print("Outputs:\n", X)

    # print("Targets:\n", targets)

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = nn.train(
        epochs, batch_size, X_train, Y_train, X_val, Y_val)

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

    input_check = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    prediction = nn.predict(input_check)
    prediction = one_hot_encode(prediction)
    target_check = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    print("Actual prediction\n", prediction)
    print("Desired prediction\n", target_check)
