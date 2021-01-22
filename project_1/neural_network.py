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
        inputs: np.array with floats

    Returns:
        np.array with floats that sum to 1 of the same shape as inputs
    """

    # We want to avoid computing e^x when x is very large, as this scales exponentially
    # Therefore, we can use a trick to "lower" all values of x, but still be able to
    # achieve the same result as discussed here
    # https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/
    # and here https://stackoverflow.com/questions/43401593/softmax-of-a-large-number-errors-out
    inputs -= max(inputs)

    denominator = sum(map(lambda x: np.exp(x), inputs))
    outputs = np.empty(inputs.shape)
    for i, x in enumerate(inputs):
        outputs[i] = np.exp(x) / denominator

    return outputs


def softmax_derivative(inputs: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: (batch size)
        outputs: outputs of model of shape: (batch size)
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape, "Targets shape: {}, outputs: {}".format(
        targets.shape, outputs.shape)

    batch_size = targets.shape
    loss = np.zeros(batch_size)
    for i in range(batch_size):
        target = targets[i]
        output = outputs[i]
        loss[i] = -(target * np.log(output) +
                    (1 - target) * np.log(1 - output))

    return np.average(loss)


def cross_entropy_loss_derivative():
    raise NotImplementedError()


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
        self.weights = np.ones((neurons_in_previous_layer, neuron_count))
        self.bias = np.ones(neuron_count)  # One bias per neuron
        self.activation_function = activation_function

    def forward_pass(self, inputs):
        a = np.transpose(self.weights).dot(inputs) + self.bias
        return run_activation_function(self.activation_function, a)

    def backward_pass(self):
        raise NotImplementedError()


class NeuralNetwork:
    def __init__(self, neurons_in_each_layer, activation_functions, loss_function, global_weight_regularization_option, global_weight_regularization_rate, initial_weight_ranges, softmax, verbose):
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
        # TODO add epochs and batches
        outputs = init_inputs
        for i, layer in enumerate(self.layers):
            if(i == 0):
                continue

            outputs = layer.forward_pass(outputs)

        print(outputs)


if __name__ == "__main__":
    pass
