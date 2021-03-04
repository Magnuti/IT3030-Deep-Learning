import numpy as np

from constants import ActivationFunction, LayerType, LossFunction
from layers import Layer, RecurrentLayer
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

    return np.mean((outputs - targets)**2)


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
    return (2.0 / number_of_outputs) * (outputs - targets)


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


# Only applicable for classification
# def accuracy(outputs, targets):
#     """
#     Calculates the accuracy (i.e., correct predictions / total predictions).

#     Args:
#         outputs: np.ndarray of shape (number_of_outputs, batch_size)
#         targets: same dimension as outputs
#     Return
#         accuracy: float
#     """
#     assert outputs.shape == targets.shape, "Shape was {}, {}".format(
#         outputs.shape, targets.shape)

#     outputs = one_hot_encode(outputs)
#     dataset_size = outputs.shape[1]
#     correct_predictions = 0
#     for i in range(dataset_size):
#         if np.all(np.equal(outputs[:, i], targets[:, i])):
#             correct_predictions += 1

#     return correct_predictions / dataset_size


class NeuralNetwork:
    def __init__(self, learning_rate, neurons_in_each_layer, layer_types, activation_functions, softmax,
                 loss_function, initial_weight_ranges, initial_bias_ranges, verbose):
        self.learning_rate = learning_rate

        if softmax:
            # Appends a SoftMax layer as the last layer with as many neurons as the last layer before SoftMax
            neurons_in_each_layer.append(neurons_in_each_layer[-1])
            layer_types.append(LayerType.DENSE)

        self.neurons_in_each_layer = neurons_in_each_layer
        self.layer_types = layer_types
        # Add None as activation function to the input layer and SoftMax to last layer
        activation_functions.insert(0, None)
        if softmax:
            activation_functions.append(ActivationFunction.SOFTMAX)
        self.activation_functions = activation_functions
        self.softmax = softmax
        self.loss_function = loss_function
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

            if self.layer_types[i - 1] == LayerType.DENSE:
                layer = Layer(neuron_count, neurons_in_previous_layer, self.activation_functions[i], self.learning_rate,
                              self.initial_weight_ranges, self.initial_bias_ranges, verbose=self.verbose, name="Dense {}".format(i))
            elif self.layer_types[i - 1] == LayerType.RECURRENT:
                layer = RecurrentLayer(neuron_count, neurons_in_previous_layer, self.activation_functions[i], self.learning_rate,
                                       self.initial_weight_ranges, self.initial_bias_ranges, verbose=self.verbose, name="Recurrent {}".format(i))
            else:
                raise NotImplementedError(self.layer_types[i - 1])

            neurons_in_previous_layer = neuron_count
            layers.append(layer)

        return layers

    def batch_loader(self, sequence_cases: list, batch_size: int, shuffle=True):
        """
        Creates a batch generator over the whole dataset (X, Y) which returns a generator iterating over all the batches.
        This function is called once each epoch.

        Args:
            sequence_cases: list(list(tuple(np.ndarray, np.ndarray)))
                (batch_size, sequence length, 2, case length)
                list of sequence-cases, where each cases is a new list that holds several tuples of input-output pairs
            shuffle (bool): To shuffle the dataset between each epoch or not.
        """
        dataset_size = len(sequence_cases)
        if dataset_size % batch_size == 0:
            num_batches = dataset_size // batch_size
        else:
            # Drop last batch if the dataset is not evenly divisible by the batch size
            num_batches = int(np.floor(dataset_size / batch_size))

        indices = list(range(dataset_size))

        if(shuffle):
            np.random.shuffle(indices)

        for i in range(num_batches):
            # Divides the indicies into into batches
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            yield np.array(sequence_cases)[batch_indices]

    def reset_layers(self):
        for layer in self.layers:
            if layer is not None:
                # Skip input layer
                layer.init_new_sequence()

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

    def backward_pass(self, R, sequence_step, last_sequence):
        # def backward_pass(self, outputs, targets, sequence_step, last_sequence):
        """
        Performs backward pass over a mini-batch.

        Args
            outputs: np.ndarray of shape (output_size, batch_size)
            targets: np.ndarray the same shape as outputs
            sequence_step: int
            last_sequence: bool
        """
        # assert outputs.shape == targets.shape

        # Jacobian loss
        # R = derivative_loss_function(self.loss_function, outputs, targets).T
        for i, layer in reversed(list(enumerate(self.layers))):
            # Skip input layer
            if layer is not None:
                R = layer.backward_pass(R, sequence_step, last_sequence)

    def train(self, epochs, batch_size, XY_train, XY_val, shuffle=True):
        """
        Performs the training phase (forward pass + backward propagation) over a number of epochs.

        Args
            epochs: int
            XY_train: list of sequence-cases
            XY_val: list of sequence-cases
            shuffle: bool, whether or not to shuffle the dataset
        """
        print("Training over {} epochs with a batch size of {}".format(
            epochs, batch_size))

        # Transpose X and Y because we want them as column vectors
        # X_val = X_val.T
        # Y_val = Y_val.T

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
                XY_train, batch_size, shuffle=shuffle)
            for sequence_cases in iter(train_loader):
                # Transpose X and Y because we want them as column vectors
                # X_batch = X_batch.T
                # Y_batch = Y_batch.T
                sequence_length = sequence_cases.shape[1]

                # Pass all cases in a sequence throuhg the network
                outputs_train = []
                losses_train = []
                for i in range(sequence_length):
                    X_batch = sequence_cases[:, i, 0, :].T
                    Y_batch = sequence_cases[:, i, 1, :].T
                    output = self.forward_pass(X_batch)
                    outputs_train.append(output)
                    loss = run_loss_function(
                        self.loss_function, output, Y_batch)
                    # print(loss)
                    losses_train.append(loss)

                loss_train = sum(losses_train)
                train_loss_history.append(loss)

                R = None
                # Backpropagate once all cases have been forwarded
                for i in range(sequence_length):
                    Y_batch = sequence_cases[:, i, 1, :].T
                    if R is None:
                        R = derivative_loss_function(
                            self.loss_function, outputs_train[i], Y_batch).T
                    else:
                        R += derivative_loss_function(
                            self.loss_function, outputs_train[i], Y_batch).T

                for i in range(sequence_length):
                    # ! Tried to use total loss derivatives
                    # TODO try with the normal way as well
                    self.backward_pass(R, i, i == sequence_length - 1)

                # loss_train = run_loss_function(
                #     self.loss_function, output_train, Y_batch)

                # Validation step at every iteration
                # if iteration % iterations_per_validation == 0:
                # output_val = self.forward_pass(X_val)

                # Must whipe caches before val
                self.reset_layers()
                XY_val = np.array(XY_val)
                losses_val = []
                for i in range(sequence_length):
                    X_batch = XY_val[:, i, 0, :].T
                    Y_batch = XY_val[:, i, 1, :].T
                    output = self.forward_pass(X_batch)
                    loss = run_loss_function(
                        self.loss_function, output, Y_batch)
                    # print(loss)
                    losses_val.append(loss)

                loss_val = sum(losses_val)

                # loss_val = run_loss_function(
                #     self.loss_function, output_val, Y_val)
                # accuracy_train = []
                # for i in range(sequence_length):
                # Y_batch = sequence_cases[:, i, 1, :].T
                # accuracy_train.append(accuracy(outputs_train[i], Y_batch))

                # accuracy_train = np.array(accuracy_train).mean()

                # accuracy_val = accuracy(output_val, Y_val)

                val_loss_history.append(loss_val)
                # train_accuracy_history.append(accuracy_train)
                # val_accuracy_history.append(accuracy_val)

                # print("Epoch: {}, iteration: {}, training loss {}:".format(
                #     epoch, iteration, loss_train))

                print("Epoch: {}, iteration: {}, training loss: {}, validation loss {}:".format(
                    epoch, iteration, loss_train, loss_val))

                iteration += 1
                # if iteration >= 3:  # !
                #     exit()

                self.reset_layers()

        return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

    def final_accuracy(self, XY_test):
        # TODO rename to loss?
        """
        Calculates the prediction for a test set

        Args
            XY_test: list(list(tuple(np.ndarray, np.ndarray)))
                list of sequence-cases, where each cases is a new list that holds several tuples of input-output pairs
        Return
            accuracy: float
        """
        XY_test = np.array(XY_test)
        sequence_length = XY_test.shape[1]
        loss = 0
        for i in range(sequence_length):
            X_batch = XY_test[:, i, 0, :].T
            Y_batch = XY_test[:, i, 1, :].T
            output = self.forward_pass(X_batch)
            loss += run_loss_function(self.loss_function, output, Y_batch)

        self.reset_layers()

        return loss

    def predict(self, X, one_hot=True):
        # TODO
        raise NotImplementedError()
        """
        Feeds X into the model and returns the output.

        Args
            X: np.ndarray of shape (batch_size, input_size)
            one_hot: bool, whether the output should be one-hot encoded.

        Return
            np.ndarray of shape (batch_size, output_size)
        """
        # self.reset_layers()
        output = self.forward_pass(X.T)
        if one_hot:
            output = one_hot_encode(output)
        return output.T


if __name__ == "__main__":
    learning_rate = 0.1
    neurons_in_each_layer = [3, 3, 3, 3]
    layer_types = [LayerType.RECURRENT, LayerType.RECURRENT, LayerType.DENSE]
    activation_functions = [
        ActivationFunction.RELU, ActivationFunction.RELU, ActivationFunction.LINEAR]
    softmax = False
    loss_function = LossFunction.MSE
    initial_weight_ranges = "glorot_normal"
    initial_bias_ranges = [0, 0]
    verbose = False
    nn = NeuralNetwork(learning_rate, neurons_in_each_layer, layer_types, activation_functions,
                       softmax, loss_function, initial_weight_ranges, initial_bias_ranges, verbose)

    def create_sequence_data():
        sequences = []
        sequences.append(
            [
                (np.array([1, 0, 0]), np.array([0, 1, 0])),
                (np.array([0, 1, 0]), np.array([0, 0, 1])),
                (np.array([0, 0, 1]), np.array([1, 0, 0])),
            ]
        )
        sequences.append(
            [
                (np.array([0, 1, 0]), np.array([1, 0, 0])),
                (np.array([1, 0, 0]), np.array([0, 0, 1])),
                (np.array([0, 0, 1]), np.array([0, 1, 0])),
            ]
        )

        return sequences

    epochs = 40
    batch_size = 2
    sequence_data = create_sequence_data()

    # XY_train, XY_val, XY_test = split_dataset(sequence_data)

    # print("Outputs:\n", X)

    # print("Targets:\n", targets)

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = nn.train(
        epochs, batch_size, sequence_data, sequence_data, shuffle=False)

    plot_loss_and_accuracy(
        train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history)

    # input_check = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    # prediction = nn.predict(input_check)
    # prediction = one_hot_encode(prediction)
    # # target_check = np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    # print("Actual prediction\n", prediction)
    # print("Desired prediction\n", target_check)
