import yaml

from constants import ActivationFunction, LossFunction, GlobalWeightRegularizationOption


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.verbose = config_data["verbose"]
        self.learning_rate = config_data["learning_rate"]
        self.batch_size = config_data["batch_size"]
        self.epochs = config_data["epochs"]
        self.neurons_in_each_layer = config_data["neurons_in_each_layer"]

        activation_functions = config_data["activation_functions"]
        self.activation_functions = []
        for i, af in enumerate(activation_functions):
            af = ActivationFunction(af)
            if(i < len(activation_functions) - 1 and af == ActivationFunction.SOFTMAX):
                raise ValueError("SoftMax can only be used at the last layer.")
            self.activation_functions.append(af)

        loss_function = config_data["loss_function"]
        self.loss_function = LossFunction(loss_function)

        global_weight_regularization_option = config_data["global_weight_regularization_option"]

        if(global_weight_regularization_option):
            self.global_weight_regularization_option = GlobalWeightRegularizationOption(
                global_weight_regularization_option)
        else:
            self.global_weight_regularization_option = None

        self.global_weight_regularization_rate = config_data["global_weight_regularization_rate"]
        self.initial_weight_ranges = config_data["initial_weight_ranges"]
        self.initial_bias_ranges = config_data["initial_bias_ranges"]
        self.dataset_filename = config_data["dataset_filename"]

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
