from enum import Enum
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math
import yaml


class Classes(Enum):
    CIRCLE = 0
    RECTANGLE = 1
    TRIANGLE = 2
    CROSS = 3


class DataGenerator:
    def __init__(self):
        self.__parse_arguments()

        if(self.image_dimension < self.min_image_dimension or self.image_dimension > self.max_image_dimension):
            raise ValueError("Invalid image dimension")

        # Create images
        circles = self.__create_circles(self.min_circle_radius)
        circles = self.__add_noise(circles)

        rectangles = self.__create_rectangles(self.min_rectangle_size)
        rectangles = self.__add_noise(rectangles)

        # Save images
        self.output_dir = pathlib.Path("images")
        self.output_dir.mkdir(exist_ok=True)

        def save_figures(figures, figure_name):
            for i, fig in enumerate(figures):
                plt.imsave(self.output_dir.joinpath(
                    "{}_{}.png".format(figure_name, i)), fig)

        save_figures(circles, "circle")
        save_figures(rectangles, "rectangle")

    def __parse_arguments(self):
        with open("data_generator_config.yaml", "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        self.training_ratio = config_data["training_ratio"]
        self.validation_ratio = config_data["validation_ratio"]
        self.testing_ratio = config_data["testing_ratio"]
        self.min_image_dimension = config_data["min_image_dimension"]
        self.max_image_dimension = config_data["max_image_dimension"]
        self.image_dimension = config_data["image_dimension"]
        self.images_in_each_class = config_data["images_in_each_class"]
        self.noise_ratio = config_data["noise_ratio"]
        self.flatten = config_data["flatten"]
        self.center = config_data["center"]
        self.min_circle_radius = config_data["min_circle_radius"]
        self.min_rectangle_size = config_data["min_rectangle_size"]

    def __add_noise(self, images):
        for i, image in enumerate(images):
            for k in range(round(self.image_dimension * self.image_dimension * self.noise_ratio)):
                x = randint(0, self.image_dimension - 1)
                y = randint(0, self.image_dimension - 1)
                images[i][y, x] = not image[y, x]

        return images

    def __create_circles(self, min_radius):
        circles = []
        for i in range(self.images_in_each_class):
            if(self.center):
                center_x = (self.image_dimension // 2)
                center_y = center_x
                if(self.image_dimension % 2 == 0):
                    max_radius = self.image_dimension / 2 - 1
                else:
                    max_radius = self.image_dimension // 2
                radius = randint(min_radius, max_radius)
            else:
                center_x = randint(
                    min_radius, self.image_dimension - min_radius - 1)
                center_y = randint(
                    min_radius, self.image_dimension - min_radius - 1)
                max_radius = min(self.image_dimension - center_x - 1,
                                 center_x, self.image_dimension - center_y - 1, center_y)
                radius = randint(min_radius, max_radius)

            img = np.zeros((self.image_dimension, self.image_dimension))

            for degree in range(0, 360, 5):
                x = round(radius * math.cos(math.radians(degree)))
                y = round(radius * math.sin(math.radians(degree)))
                img[center_y + y, center_x + x] = 1

            circles.append(img)

        return circles

    def __create_rectangles(self, min_rectangle_size):
        rectangles = []
        for i in range(self.images_in_each_class):
            if(self.center):
                center_x = (self.image_dimension // 2)
                center_y = center_x
                if(self.image_dimension % 2 == 0):
                    max_height = self.image_dimension / 2 - 1
                    max_width = max_height
                else:
                    max_height = self.image_dimension // 2
                    max_width = max_height
                height = randint(min_rectangle_size, max_height)
                width = randint(min_rectangle_size, max_width)

                start_y = center_y - height
                end_y = center_y + height
                start_x = center_x - width
                end_x = center_x + width
            else:
                start_x = randint(0, self.image_dimension -
                                  min_rectangle_size - 1)
                start_y = randint(0, self.image_dimension -
                                  min_rectangle_size - 1)
                end_y = randint(start_y + min_rectangle_size,
                                self.image_dimension - 1)
                end_x = randint(start_x + min_rectangle_size,
                                self.image_dimension - 1)

            img = np.zeros((self.image_dimension, self.image_dimension))

            img[start_y:end_y, start_x] = 1
            img[start_y:end_y, end_x] = 1
            img[start_y, start_x:end_x] = 1
            img[end_y, start_x:end_x] = 1

            rectangles.append(img)

        return rectangles

    def __create_triangles(self):
        raise NotImplementedError()

    def __create_crosses(self):
        raise NotImplementedError()


if __name__ == "__main__":
    data_generator = DataGenerator()
