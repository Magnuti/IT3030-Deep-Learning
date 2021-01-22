import yaml

from constants import ActivationFunction, LossFunction, GlobalWeightRegularizationOption


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        neurons_in_each_layer = config_data["neurons_in_each_layer"]
        activation_functions = config_data["activation_functions"]
        loss_function = config_data["loss_function"]
        global_weight_regularization_option = config_data["global_weight_regularization_option"]
        global_weight_regularization_rate = config_data["global_weight_regularization_rate"]
        initial_weight_ranges = config_data["initial_weight_ranges"]
        softmax = config_data["softmax"]
        dataset_filename = config_data["dataset_filename"]
        verbose = config_data["verbose"]

        self.neurons_in_each_layer = neurons_in_each_layer

        self.activation_functions = []
        for i, af in enumerate(activation_functions):
            af = ActivationFunction(af)
            if(i < len(activation_functions) - 1 and af == ActivationFunction.SOFTMAX):
                raise ValueError("SoftMax can only be used at the last layer.")
            self.activation_functions.append(af)

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
        self.verbose = verbose

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
