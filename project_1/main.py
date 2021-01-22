import numpy as np

from config_parser import Arguments
from neural_network import NeuralNetwork

if __name__ == "__main__":
    args = Arguments()
    args.parse_arguments()

    nn = NeuralNetwork(args.neurons_in_each_layer, args.activation_functions, args.loss_function, args.global_weight_regularization_option,
                       args.global_weight_regularization_rate, args.initial_weight_ranges, args.softmax, args.verbose)

    img = np.ones((25, 25))
    img = img.flatten()

    nn.train(1, img)
