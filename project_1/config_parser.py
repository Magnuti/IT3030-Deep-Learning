import yaml

from constants import ActivationFunction, LossFunction, GlobalWeightRegularizationOption


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        hidden_layers = config_data["hidden_layers"]
        neurons_in_each_layer = config_data["neurons_in_each_layer"]
        activation_function = config_data["activation_function"]
        loss_function = config_data["loss_function"]
        global_weight_regularization_option = config_data["global_weight_regularization_option"]
        global_weight_regularization_rate = config_data["global_weight_regularization_rate"]
        initial_weight_ranges = config_data["initial_weight_ranges"]
        softmax = config_data["softmax"]
        dataset_filename = config_data["dataset_filename"]

        self.hidden_layers = hidden_layers
        self.neurons_in_each_layer = neurons_in_each_layer

        self.activation_function = []
        for af in activation_function:
            self.activation_function.append(ActivationFunction(af))

        self.loss_function = LossFunction(loss_function)

        if(global_weight_regularization_option):
            self.global_weight_regularization_option = GlobalWeightRegularizationOption(
                global_weight_regularization_option)
        else:
            self.global_weight_regularization_option = None

        self.global_weight_regularization_rate = global_weight_regularization_rate
        self.initial_weight_ranges = initial_weight_ranges
        self.softmax = softmax
        self.dataset_filename = dataset_filename

    def __str__(self):
        x = "Arguments: {\n"
        for key, value in self.__dict__.items():
            x += "\t{}: {}\n".format(key, value)
        x += "}"
        return x


if __name__ == "__main__":
    arguments = Arguments()
    arguments.parse_arguments()
    print(arguments)
