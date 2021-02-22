import yaml


class DataGeneratorArguments:
    def parse_arguments(self):
        with open("data_generator_config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.pattern_length = config_data["pattern_length"]
        self.sequence_length = config_data["sequence_length"]
        self.sequences = config_data["sequences"]
        self.noise_ratio = config_data["noise_ratio"]

    def __str__(self):
        x = "Arguments: {\n"
        for key, value in self.__dict__.items():
            x += "\t{}: {}\n".format(key, value)
        x += "}"
        return x
