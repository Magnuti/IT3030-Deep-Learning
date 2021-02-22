import numpy as np

from data_generator_parser import DataGeneratorArguments
from data_generator import DataGenerator
from config_parser import Arguments
from neural_network import NeuralNetwork
from utils import split_dataset, plot_loss_and_accuracy

if __name__ == "__main__":
    data_generator_args = DataGeneratorArguments()
    data_generator_args.parse_arguments()

    nn_args = Arguments()
    nn_args.parse_arguments()

    data_generator = DataGenerator(data_generator_args)
    sequence_cases = data_generator.get_sequences(shuffle=False)

    nn = NeuralNetwork(nn_args.learning_rate, nn_args.neurons_in_each_layer, nn_args.layer_types, nn_args.activation_functions,
                       nn_args.softmax, nn_args.loss_function, nn_args.initial_weight_ranges, nn_args.initial_bias_ranges, nn_args.verbose)

    XY_train, XY_val, XY_test = split_dataset(
        sequence_cases, nn_args.training_ratio, nn_args.validation_ratio, nn_args.testing_ratio)

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = nn.train(
        nn_args.epochs, nn_args.batch_size, XY_train, XY_val)

    accuracy = nn.final_accuracy(XY_test)
    print("Final accuracy", accuracy)

    plot_loss_and_accuracy(
        train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history)
