from enum import Enum
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math

from data_generator_parser import DataGeneratorArguments


class Classes(Enum):
    CIRCLE = 0
    RECTANGLE = 1
    TRIANGLE = 2
    CROSS = 3


class DataGenerator:
    def __init__(self, args):
        self.args = args

        self.output_dir = pathlib.Path("images")
        self.output_dir.mkdir(exist_ok=True)

    def create_figures(self):
        # TODO add args.flatten option to return either 2D-array or 1D array

        circles = self.__create_circles(self.args.circle_radius_range)

        rectangles = self.__create_rectangles(
            self.args.rectanlge_range_height, self.args.rectanlge_range_width)

        crosses = self.__create_crosses(
            self.args.cross_size_range, self.args.cross_thickness_range)

        # Add noise
        circles = self.__add_noise(circles)
        rectangles = self.__add_noise(rectangles)
        crosses = self.__add_noise(crosses)

        def save_figures(figures, figure_name):
            for i, fig in enumerate(figures):
                plt.imsave(self.output_dir.joinpath(
                    "{}_{}.png".format(figure_name, i)), fig)

        # Save images
        save_figures(circles, "circle")
        save_figures(rectangles, "rectangle")
        save_figures(crosses, "cross")

    def __add_noise(self, images):
        for i, image in enumerate(images):
            for k in range(round(self.args.image_dimension * self.args.image_dimension * self.args.noise_ratio)):
                x = randint(0, self.args.image_dimension - 1)
                y = randint(0, self.args.image_dimension - 1)
                images[i][y, x] = not image[y, x]

        return images

    def __create_circles(self, circle_radius_range):
        circles = []
        for i in range(self.args.images_in_each_class):
            if(self.args.center):
                center_x = (self.args.image_dimension // 2)
                center_y = center_x
                if(self.args.image_dimension % 2 == 0):
                    max_radius = self.args.image_dimension / 2 - 1
                else:
                    max_radius = self.args.image_dimension // 2
                min_radius = circle_radius_range[0]
                max_radius = min(max_radius, circle_radius_range[1])
                radius = randint(min_radius, max_radius)
            else:
                center_x = randint(
                    circle_radius_range[0], self.args.image_dimension - circle_radius_range[0] - 1)
                center_y = randint(
                    circle_radius_range[0], self.args.image_dimension - circle_radius_range[0] - 1)
                max_radius = min(self.args.image_dimension - center_x - 1,
                                 center_x, self.args.image_dimension - center_y - 1, center_y)
                max_radius = min(max_radius, circle_radius_range[1])
                radius = randint(circle_radius_range[0], max_radius)

            img = np.zeros((self.args.image_dimension,
                            self.args.image_dimension))

            for degree in range(0, 360, 5):
                x = round(radius * math.cos(math.radians(degree)))
                y = round(radius * math.sin(math.radians(degree)))
                img[center_y + y, center_x + x] = 1

            circles.append(img)

        return circles

    def __create_rectangles(self, rectanlge_range_height, rectanlge_range_width):
        rectangles = []
        for i in range(self.args.images_in_each_class):
            if(self.args.center):
                center_x = (self.args.image_dimension // 2)
                center_y = center_x
                if(self.args.image_dimension % 2 == 0):
                    max_half_height = self.args.image_dimension / 2 - 1
                    max_half_width = max_half_height
                else:
                    max_half_height = self.args.image_dimension // 2
                    max_half_width = max_half_height

                max_half_height = min(
                    max_half_height, rectanlge_range_height[1] // 2)
                max_half_width = min(
                    max_half_width, rectanlge_range_width[1] // 2)

                height = randint(
                    rectanlge_range_height[0] // 2, max_half_height)
                width = randint(rectanlge_range_width[0] // 2, max_half_width)

                start_y = center_y - height
                end_y = center_y + height
                start_x = center_x - width
                end_x = center_x + width
            else:
                start_y = randint(0, self.args.image_dimension -
                                  rectanlge_range_height[0] - 1)
                start_x = randint(0, self.args.image_dimension -
                                  rectanlge_range_width[0] - 1)

                max_y = min(self.args.image_dimension - 1,
                            start_y + rectanlge_range_height[1])
                max_x = min(self.args.image_dimension - 1,
                            start_x + rectanlge_range_width[1])

                end_y = randint(start_y + rectanlge_range_height[0], max_y)
                end_x = randint(start_x + rectanlge_range_width[0], max_x)

            img = np.zeros((self.args.image_dimension,
                            self.args.image_dimension))

            img[start_y:end_y, start_x] = 1
            img[start_y:end_y, end_x] = 1
            img[start_y, start_x:end_x] = 1
            img[end_y, start_x:end_x] = 1

            rectangles.append(img)

        return rectangles

    def __create_triangles(self):
        raise NotImplementedError()

    def __create_crosses(self, cross_size_range, cross_thickness_range):
        crosses = []
        for i in range(self.args.images_in_each_class):
            size = randint(cross_size_range[0], min(
                cross_size_range[1], self.args.image_dimension))

            if(self.args.center):
                center_y = self.args.image_dimension // 2
                center_x = center_y
            else:
                center_y = randint(
                    size // 2, self.args.image_dimension - size // 2)
                center_x = randint(
                    size // 2, self.args.image_dimension - size // 2)

            img = np.zeros((self.args.image_dimension,
                            self.args.image_dimension))

            thickness = randint(
                cross_thickness_range[0], cross_thickness_range[1])
            size_1 = math.floor(size / 2)
            size_2 = math.ceil(size / 2)
            thickness_1 = math.floor(thickness / 2)
            thickness_2 = math.ceil(thickness / 2)
            img[center_y - size_1:center_y + size_2, center_x -
                thickness_1: center_x + thickness_2] = 1
            img[center_y - thickness_1:center_y + thickness_2,
                center_x - size_1: center_x + size_2] = 1

            crosses.append(img)

        return crosses


if __name__ == "__main__":
    args = DataGeneratorArguments()
    args.parse_arguments()
    data_generator = DataGenerator(args)
    data_generator.create_figures()
