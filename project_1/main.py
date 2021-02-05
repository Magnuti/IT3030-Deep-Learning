import numpy as np

from data_generator_parser import DataGeneratorArguments
from config_parser import Arguments
from neural_network import NeuralNetwork

if __name__ == "__main__":
    data_generator_args = DataGeneratorArguments()
    data_generator_args.parse_arguments()

    args = Arguments()
    args.parse_arguments()

    nn = NeuralNetwork(args.learning_rate, args.batch_size, args.neurons_in_each_layer, args.activation_functions, args.loss_function,
                       args.global_weight_regularization_option, args.global_weight_regularization_rate, args.initial_weight_ranges, args.softmax, args.verbose)

    image_size = data_generator_args.image_dimension**2
    img = np.ones(
        image_size * args.batch_size).reshape((image_size, args.batch_size))

    nn.train(1, img)
