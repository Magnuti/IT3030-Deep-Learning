import numpy as np

from constants import ActivationFunction


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


def softmax(x: np.ndarray) -> np.ndarray:
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


def run_activation_function(af, X):
    """
    Args:
        X: np.ndarray of shape (neurons, batch_size)

    Returns
        np.ndarray of the same shape as X
    """
    if(af == ActivationFunction.SIGMOID):
        return sigmoid(X)
    elif(af == ActivationFunction.TANH):
        return tanh(X)
    elif(af == ActivationFunction.RELU):
        return relu(X)
    elif(af == ActivationFunction.LINEAR):
        return linear(X)
    elif(af == ActivationFunction.SOFTMAX):
        return softmax(X)
    else:
        raise NotImplementedError()


def derivative_activation_function(af, X):
    """
    Args:
        X: np.ndarray of shape (neurons, batch_size)

    Returns
        np.ndarray of the same shape as X
    """
    if(af == ActivationFunction.SIGMOID):
        return sigmoid_derivative(X)
    elif(af == ActivationFunction.TANH):
        return tanh_derivative(X)
    elif(af == ActivationFunction.RELU):
        return relu_derivative(X)
    elif(af == ActivationFunction.LINEAR):
        return linear_derivative(X)
    # SoftMax derivative is not used
    else:
        raise NotImplementedError()


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
        self.inputs = X

        if self.activation_function == ActivationFunction.SOFTMAX:
            weighted_inputs = X
        else:
            weighted_inputs = np.matmul(self.weights.T, X) + self.bias

        # self.activations has shape (neurons, batch_size)
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
        """
        Performs backward pass over one layer.

        Args
            R: np.ndarray of shape (batch_size, neurons_in_this_layer)

        Returns
            np.ndarray of shape (batch_size, neurons_in_previous_layer), where neurons_in_previous_layer is
            the neuron count of the layer to the left (i.e., the input to this layer).
        """
        if self.activation_function == ActivationFunction.SOFTMAX:
            activations = self.activations.T  # (batch_size, neurons)

            batch_size = activations.shape[0]
            for b in range(batch_size):
                # Builds the J-Soft matrix for each case in the batch
                j_soft = np.empty((self.neuron_count, self.neuron_count))
                for i in range(self.neuron_count):
                    for j in range(self.neuron_count):
                        if i == j:
                            j_soft[i, j] = activations[b, i] - \
                                activations[b, i] ** 2
                        else:
                            j_soft[i, j] = - activations[b, i] * \
                                activations[b, j]

                # Multiply iteratively because j_soft changes for each case in the batch
                R[b] = np.matmul(R[b], j_soft)

            return R

        activation_gradient = derivative_activation_function(
            self.activation_function, self.activations).T
        R *= activation_gradient

        # Gradients for weights and bias
        batch_size = R.shape[0]
        # Divide by batch_size to get the average gradients over the batch
        # The average works because matrix multiplication sums the gradients
        gradient_weights = np.matmul(self.inputs, R) / batch_size
        gradient_bias = R.sum(axis=0, keepdims=True).T / batch_size

        self.weights -= self.learning_rate * gradient_weights
        self.bias -= self.learning_rate * gradient_bias

        return np.matmul(R, self.weights.T)

    def __str__(self):
        return "{} neurons with {} as activation function".format(self.neuron_count, self.activation_function)


if __name__ == "__main__":
    pass
