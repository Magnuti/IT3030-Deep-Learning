import yaml

from constants import Dataset


class Arguments:
    def parse_arguments(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.dataset = Dataset(config_data["dataset"])
        self.loss_function_auto_encoder = config_data["loss_function_auto_encoder"]
        self.loss_function_classifier = config_data["loss_function_classifier"]
        self.optimizer_autoencoder = config_data["optimizer_autoencoder"]
        self.optimizer_classifier = config_data["optimizer_classifier"]
        self.learning_rate_auto_encoder = config_data["learning_rate_auto_encoder"]
        self.learning_rate_classifier = config_data["learning_rate_classifier"]
        self.latent_vector_size = config_data["latent_vector_size"]
        self.epochs_auto_encoder = config_data["epochs_auto_encoder"]
        self.epochs_classifier = config_data["epochs_classifier"]
        self.batch_size = config_data["batch_size"]
        self.labeled_to_unlabeled_split_ratio = config_data["labeled_to_unlabeled_split_ratio"]
        self.train_to_test_ratio = config_data["train_to_test_ratio"]
        self.validation_ratio = config_data["validation_ratio"]
        self.freeze = config_data["freeze"]
        self.visualize = config_data["visualize"]
        self.auto_encoder_reconstructions = config_data["auto_encoder_reconstructions"]
        self.latent_vector_plot_count = config_data["latent_vector_plot_count"]

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
