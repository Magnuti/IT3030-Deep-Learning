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

        self.init_new_sequence()

    def forward_pass(self, X):
        """
        Args:
            X: np.ndarray of shape (self.neurons_in_previous_layer, batch_size)
        Returns:
            np.ndarray of shape (self.neuron_count, batch_size)
        """
        self.inputs_history.append(X)

        if self.activation_function == ActivationFunction.SOFTMAX:
            weighted_inputs = X
        else:
            weighted_inputs = np.matmul(self.weights.T, X) + self.bias

        # activations has shape (neurons, batch_size)
        activations = run_activation_function(
            self.activation_function, weighted_inputs)

        self.activations_history.append(activations)

        if self.verbose:
            print("Forward passing layer", self.name)
            print("Input (input_size, batch_size):\n", X)
            print("Weights:\n", self.weights)
            print("Bias:\n", self.bias)
            print("Output:\n", activations)

        return activations

    def backward_pass(self, R, sequence_step, last_sequence):
        """
        Performs backward pass over one layer.

        Args
            R: np.ndarray of shape (batch_size, neurons_in_this_layer)
            TODO

        Returns
            np.ndarray of shape (batch_size, neurons_in_previous_layer), where neurons_in_previous_layer is
            the neuron count of the layer to the left (i.e., the input to this layer).
        """
        if self.activation_function == ActivationFunction.SOFTMAX:
            # (batch_size, neurons_in_this_layer)
            activations = self.activations_history[sequence_step].T

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

        # (batch_size, neurons_in_this_layer)
        activation_gradients = derivative_activation_function(
            self.activation_function, self.activations_history[sequence_step]).T
        R *= activation_gradients

        # Gradients for weights and bias
        batch_size = R.shape[0]
        # Divide by batch_size to get the average gradients over the batch
        # The average works because matrix multiplication sums the gradients

        # (neurons_in_previous_layer, batch_size) @ (batch_size, neurons_in_this_layer)
        gradient_weights = np.matmul(
            self.inputs_history[sequence_step], R) / batch_size
        gradient_bias = R.sum(axis=0, keepdims=True).T / batch_size

        if self.gradient_weights is None:
            self.gradient_weights = gradient_weights
        else:
            self.gradient_weights += gradient_weights  # Accumulate the gradients as we go

        if self.gradient_bias is None:
            self.gradient_bias = gradient_bias
        else:
            self.gradient_bias += gradient_bias  # Accumulate the gradients as we go

        if last_sequence:
            # Update parameters on the last sequence
            self.weights -= self.learning_rate * self.gradient_weights
            self.bias -= self.learning_rate * self.gradient_bias

        # (batch_size, neurons_in_this_layer) @ (neurons_in_this_layer, neurons_in_previous_layer)
        return np.matmul(R, self.weights.T)

    def init_new_sequence(self):
        # Cache all activations and inputs because we need it to do backpropagation
        self.inputs_history = []
        self.activations_history = []
        self.gradient_weights = None
        self.gradient_bias = None

    def __str__(self):
        return "{} neurons with {} as activation function".format(self.neuron_count, self.activation_function)


class RecurrentLayer(Layer):
    def __init__(self, neuron_count, neurons_in_previous_layer, activation_function, learning_rate,
                 initial_weight_ranges, initial_bias_ranges, verbose=False, name=""):
        super().__init__(neuron_count, neurons_in_previous_layer, activation_function,
                         learning_rate, initial_weight_ranges, initial_bias_ranges, verbose, name)

        # Init the recurrent weights in the same fashion as the "normal" weights.
        if initial_weight_ranges == "glorot_normal":
            self.recurrent_weights = glorot_normal(
                neuron_count, neuron_count)
        elif initial_weight_ranges == "glorot_uniform":
            self.recurrent_weights = glorot_uniform(
                neuron_count, neuron_count)
        else:
            self.recurrent_weights = init_weights_with_range(
                initial_weight_ranges[0], initial_weight_ranges[1], neuron_count, neuron_count)

        self.output_jacobian = None

    def forward_pass(self, X):
        """
        Args:
            X: np.ndarray of shape (self.neurons_in_previous_layer, batch_size)
        Returns:
            np.ndarray of shape (self.neuron_count, batch_size)
        """
        self.inputs_history.append(X)

        weighted_inputs = np.matmul(self.weights.T, X) + self.bias

        # Skip first forward in a sequence
        if self.activations_history:
            weighted_inputs += np.matmul(self.recurrent_weights.T,
                                         self.activations_history[-1])

        # activations has shape (neurons, batch_size)
        activations = run_activation_function(
            self.activation_function, weighted_inputs)

        self.activations_history.append(activations)

        if self.verbose:
            print("Forward passing layer", self.name)
            print("Input (input_size, batch_size):\n", X)
            print("Weights:\n", self.weights)
            print("Bias:\n", self.bias)
            print("Output:\n", activations)

        return activations

    def backward_pass(self, R, sequence_step, last_sequence):
        """
        Performs backward pass over one layer.

        Args
            R: np.ndarray of shape (batch_size, neurons_in_this_layer)
            sequence_step: int
            last_sequence: bool
        Returns
            np.ndarray of shape (batch_size, neurons_in_previous_layer), where neurons_in_previous_layer is
            the neuron count of the layer to the left (i.e., the input to this layer).
        """
        batch_size = R.shape[0]
        # MxM matrix which is fully connected to itself. Thus, M = neurons_in_this_layer
        # Page 18-20 in the slides

        # Save the output Jacobian as we need it in the recurrent sequence
        if self.output_jacobian is None:
            # Treated as a normal dense layer
            self.output_jacobian = R
        else:
            # (batch_size, neurons_in_this_layer)
            activation_gradients = derivative_activation_function(
                self.activation_function, self.activations_history[sequence_step]).T
            # ! Shouldn't we derive the activation function here?
            # x = self.activations_history[sequence_step].T # ?
            x = activation_gradients  # ?

            # (batch_size, neurons_in_this_layer, neurons_in_this_layer)
            diag_matrix = np.empty((batch_size, x.shape[1], x.shape[1]))
            for batch in range(batch_size):
                diag_matrix[batch] = np.diag(x[batch])

            # (batch_size, neurons_in_this_layer, neurons_in_this_layer) @ (neurons_in_this_layer, neurons_in_this_layer)
            recurrent_jacobian = np.matmul(
                diag_matrix, self.recurrent_weights.T)

            assert recurrent_jacobian.shape == (
                batch_size, self.neuron_count, self.neuron_count)

            part_2 = np.empty_like(R)
            for i in range(batch_size):
                output_jacobian = self.output_jacobian[i]
                output_jacobian = output_jacobian.reshape(
                    (1, output_jacobian.shape[0]))

                part_2[i] = np.matmul(output_jacobian, recurrent_jacobian[i])

            # Page 17 in the slides
            R += part_2
            self.output_jacobian = R  # (batch_size, neurons_in_this_layer)

        # (batch_size, neurons_in_this_layer)
        activation_gradients = derivative_activation_function(
            self.activation_function, self.activations_history[sequence_step]).T
        R *= activation_gradients

        # Gradients for weights and bias

        # Divide by batch_size to get the average gradients over the batch
        # The average works because matrix multiplication sums the gradients
        gradient_weights = np.matmul(
            self.inputs_history[sequence_step], R) / batch_size
        gradient_bias = R.sum(axis=0, keepdims=True).T / batch_size

        # TODO may be moved up/down, read up on this (before or after R update)
        if self.gradient_weights is None:
            self.gradient_weights = gradient_weights
        else:
            self.gradient_weights += gradient_weights  # Accumulate the gradients as we go

        if self.gradient_bias is None:
            self.gradient_bias = gradient_bias
        else:
            self.gradient_bias += gradient_bias  # Accumulate the gradients as we go

        if last_sequence:
            # Update parameters on the last sequence
            self.weights -= self.learning_rate * self.gradient_weights
            self.bias -= self.learning_rate * self.gradient_bias

        # assert self.output_jacobian.shape == (batch_size, self.neurons_in_previous_layer), "Expected {}, got {}".format(
        #     (batch_size, self.neurons_in_previous_layer), self.output_jacobian.shape)

        return np.matmul(R, self.weights.T)

    def init_new_sequence(self):
        super().init_new_sequence()
        self.output_jacobian = None


if __name__ == "__main__":
    pass
