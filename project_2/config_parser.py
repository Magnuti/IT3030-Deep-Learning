import yaml

from constants import ActivationFunction, LayerType, LossFunction


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.training_ratio = config_data["training_ratio"]
        self.validation_ratio = config_data["validation_ratio"]
        self.testing_ratio = config_data["testing_ratio"]
        self.verbose = config_data["verbose"]
        self.learning_rate = config_data["learning_rate"]
        self.batch_size = config_data["batch_size"]
        self.epochs = config_data["epochs"]
        self.neurons_in_each_layer = config_data["neurons_in_each_layer"]

        layer_types = config_data["layer_types"]
        self.layer_types = []
        for layer in layer_types:
            self.layer_types.append(LayerType(layer))

        activation_functions = config_data["activation_functions"]
        self.activation_functions = []
        for i, af in enumerate(activation_functions):
            self.activation_functions.append(ActivationFunction(af))

        self.softmax = config_data["softmax"]
        self.loss_function = LossFunction(config_data["loss_function"])
        self.initial_weight_ranges = config_data["initial_weight_ranges"]
        self.initial_bias_ranges = config_data["initial_bias_ranges"]

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
