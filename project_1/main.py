import numpy as np

from data_generator_parser import DataGeneratorArguments
from data_generator import DataGenerator
from config_parser import Arguments
from neural_network import NeuralNetwork
from utils import split_dataset, plot_loss_and_accuracy, shuffle_data_and_targets

if __name__ == "__main__":
    data_generator_args = DataGeneratorArguments()
    data_generator_args.parse_arguments()

    nn_args = Arguments()
    nn_args.parse_arguments()

    data_generator = DataGenerator(data_generator_args)
    data, targets = data_generator.get_images()
    data, targets = shuffle_data_and_targets(data, targets)

    nn = NeuralNetwork(nn_args.learning_rate, nn_args.neurons_in_each_layer, nn_args.activation_functions, nn_args.softmax, nn_args.loss_function,
                       nn_args.global_weight_regularization_option, nn_args.global_weight_regularization_rate, nn_args.initial_weight_ranges,
                       nn_args.initial_bias_ranges, nn_args.verbose)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(
        data, targets)

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = nn.train(
        nn_args.epochs, nn_args.batch_size, X_train, Y_train, X_val, Y_val)

    plot_loss_and_accuracy(train_loss_history, train_accuracy_history,
               val_loss_history, val_accuracy_history)
