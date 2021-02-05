import numpy as np

from constants import ActivationFunction


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
        return softmax(inputs)
    else:
        raise NotImplementedError()


def sigmoid(inputs: np.ndarray) -> np.ndarray:
    return 1 / 1 + np.exp(-inputs)


def sigmoid_derivative(inputs: np.ndarray) -> np.ndarray:
    # return x * (1 - x)
    raise NotImplementedError()


def tanh(inputs: np.ndarray) -> np.ndarray:
    return np.tanh(inputs)


def tanh_derivative(inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


def relu(inputs):
    inputs[inputs <= 0] = 0
    return inputs


def relu_derivative(inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError()
    # if(inputs > 0):
    #     return 1
    # return 0


def linear(inputs: np.ndarray) -> np.ndarray:
    return inputs


def linear_derivative(inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError()
    # return 1


def softmax(inputs: np.ndarray) -> np.ndarray:
    """
    Args:
        inputs: np.array of shape(num_classes, batch_size)

    Returns:
        np.array that sum to 1 of the same shape as inputs
    """

    # We want to avoid computing e^x when x is very large, as this scales exponentially
    # Therefore, we can use a trick to "lower" all values of x, but still be able to
    # achieve the same result as discussed here
    # https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
    # and here https://stackoverflow.com/questions/43401593/softmax-of-a-large-number-errors-out
    # We do this by subtracting the max values for each batch.
    inputs -= np.amax(inputs, axis=0)

    inputs = np.exp(inputs)

    return inputs / inputs.sum(axis=0)


def softmax_derivative(inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: (batch size, num_classes)
        outputs: outputs of model of shape: (batch size, num_classes)
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = - np.sum(targets * np.log(outputs), axis=1)
    return np.average(loss)


def cross_entropy_loss_derivative(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: (batch size, num_classes)
        outputs: outputs of model of shape: (batch size, num_classes)
    Returns:
        Cross entropy error (float)
    """

    # TODO add support for batches
    return np.where(predictions != 0, -targets/predictions, 0.0)

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


def binary_cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: (batch size)
        outputs: outputs of model of shape: (batch size)
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


def MSE(targets: np.ndarray, outputs: np.ndarray) -> float:
    raise NotImplementedError()
    # """
    # Args:
    #     targets: labels/targets of each image of shape: (batch size)
    #     outputs: outputs of model of shape: (batch size, number of cases)
    # Returns:
    #     Mean squared error (float)
    # """

    # batch_size = outputs.shape[0]
    # loss = np.zeros(batch_size)
    # for i in range(batch_size):
    #     target = targets[i]
    #     output = outputs[i]
    #     loss[i] = (output - target)**2

    # return np.average(loss)


def MSE_derivative():
    # See slides for lecture 1 page 5
    # This must be calculated for every weight
    raise NotImplementedError()


class Layer:
    def __init__(self, neuron_count, neurons_in_previous_layer, activation_function):
        self.neurons = neuron_count
        # TODO set weights to a low value around the normal distribution
        self.weights = np.zeros((neurons_in_previous_layer, neuron_count))
        # One bias per neuron, column vector
        self.bias = np.ones((neuron_count, 1))  # TODO "often init as zero"
        self.activation_function = activation_function

    def forward_pass(self, inputs):
        """
        Args:
            inputs: np.ndarray of shape (self.neurons_in_previous_layer, batch_size)
        Returns:
            np.ndarray of shape (self.neuron_count, batch_size)
        """
        a = np.matmul(self.weights.T, inputs) + self.bias
        return run_activation_function(self.activation_function, a)

    def backward_pass(self):
        raise NotImplementedError()


class NeuralNetwork:
    def __init__(self, learning_rate, batch_size, neurons_in_each_layer, activation_functions,
                 loss_function, global_weight_regularization_option,
                 global_weight_regularization_rate, initial_weight_ranges, softmax, verbose):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.neurons_in_each_layer = neurons_in_each_layer
        # Add None as af to the input layer
        activation_functions.insert(0, None)
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.global_weight_regularization_option = global_weight_regularization_option
        self.global_weight_regularization_rate = global_weight_regularization_rate
        self.initial_weight_ranges = initial_weight_ranges
        self.softmax = softmax
        self.verbose = verbose

        self.layers = self.__build_network()

        if(self.verbose):
            for i, layer in enumerate(self.layers):
                print("Layer {}: {} neurons with {} as activation function".format(
                    i, layer.neurons, self.activation_functions[i]))

    def __build_network(self):
        layers = []
        neurons_in_previous_layer = 0
        for i, neuron_count in enumerate(self.neurons_in_each_layer):
            layers.append(
                Layer(neuron_count, neurons_in_previous_layer, self.activation_functions[i]))
            neurons_in_previous_layer = neuron_count

        return layers

    def train(self, epochs, init_inputs):
        print("Input:", init_inputs.shape)
        # TODO add epochs and batches
        outputs = init_inputs
        for i, layer in enumerate(self.layers):
            if(i == 0):
                continue

            outputs = layer.forward_pass(outputs)

        print(outputs)


if __name__ == "__main__":
    pass
