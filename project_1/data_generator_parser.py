import yaml


class DataGeneratorArguments:
    def parse_arguments(self):
        with open("data_generator_config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.training_ratio = config_data["training_ratio"]
        self.validation_ratio = config_data["validation_ratio"]
        self.testing_ratio = config_data["testing_ratio"]
        self.min_image_dimension = config_data["min_image_dimension"]
        self.max_image_dimension = config_data["max_image_dimension"]
        self.image_dimension = config_data["image_dimension"]

        if(self.image_dimension < self.min_image_dimension or self.image_dimension > self.max_image_dimension):
            raise ValueError("Invalid image dimension")

        self.circle_radius_range = config_data["circle_radius_range"]
        self.rectanlge_range_height = config_data["rectanlge_range_height"]
        self.rectanlge_range_width = config_data["rectanlge_range_width"]
        self.triangle_range = config_data["triangle_range"]
        self.cross_size_range = config_data["cross_size_range"]
        self.cross_thickness_range = config_data["cross_thickness_range"]
        self.images_in_each_class = config_data["images_in_each_class"]
        self.noise_ratio = config_data["noise_ratio"]
        self.flatten = config_data["flatten"]
        self.center = config_data["center"]

    def __str__(self):
        x = "Arguments: {\n"
        for key, value in self.__dict__.items():
            x += "\t{}: {}\n".format(key, value)
        x += "}"
        return x
