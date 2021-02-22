from enum import Enum
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math
from os import listdir

from data_generator_parser import DataGeneratorArguments


class Classes(Enum):
    CIRCLE = 0
    RECTANGLE = 1
    VERTICAL_BARS = 2
    CROSS = 3


class DataGenerator:
    def __init__(self, args):
        self.args = args

        self.output_dir = pathlib.Path("images")
        self.output_dir.mkdir(exist_ok=True)

    def get_images(self, flatten=True):
        """
        Returns data and targets.
        """
        class_names = [e.name.lower() for e in Classes]
        files = listdir(self.output_dir)
        data = []
        targets = []
        for i, f in enumerate(files):
            for i, class_name in enumerate(class_names):
                if class_name in f:
                    target_number = i
                    break

            target = np.zeros(len(class_names))
            target[target_number] = 1.0

            image = plt.imread(self.output_dir.joinpath(f))
            # Apparently this returns a (M, N, 4) array because it thinks it is a  RGBA image..
            image = image[:, :, 0]

            if flatten:
                image = image.flatten()

            data.append(image)
            targets.append(target)

        return np.array(data), np.array(targets)

    def random_sample(self, number_of_images=10):
        files = listdir(self.output_dir)
        plt.subplots(len(Classes), number_of_images)
        for i, figure_class in enumerate(Classes):
            figure_name = figure_class.name.lower()
            figure_files = list(
                filter(lambda x: figure_name in x, files))
            figure_files = np.random.choice(
                figure_files, size=number_of_images)
            for j, f in enumerate(figure_files):
                image = plt.imread(self.output_dir.joinpath(f))
                # Apparently this returns a (M, N, 4) array because it thinks it is a  RGBA image..
                image = image[:, :, 0]

                plt.subplot(len(Classes), number_of_images,
                            i * number_of_images + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(image)

        plt.show()

    def create_figures(self):

        circles = self.__create_circles(self.args.circle_radius_range)

        rectangles = self.__create_rectangles(
            self.args.rectanlge_range_height, self.args.rectanlge_range_width)

        vertical_bars = self.__create_vertical_bars(
            self.args.vertical_bar_width)

        crosses = self.__create_crosses(
            self.args.cross_size_range, self.args.cross_thickness_range)

        # Add noise
        circles = self.__add_noise(circles)
        rectangles = self.__add_noise(rectangles)
        vertical_bars = self.__add_noise(vertical_bars)
        crosses = self.__add_noise(crosses)

        def save_figures(figures, figure_name):
            for i, fig in enumerate(figures):
                fig *= 255
                plt.imsave(self.output_dir.joinpath(
                    "{}_{}.png".format(figure_name, i)), fig, cmap="gray")

        # Save images
        save_figures(circles, Classes.CIRCLE.name.lower())
        save_figures(rectangles, Classes.RECTANGLE.name.lower())
        save_figures(vertical_bars, Classes.VERTICAL_BARS.name.lower())
        save_figures(crosses, Classes.CROSS.name.lower())

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

            img[end_y, end_x] = 1  # Hotfix for the corner pixel bottom-right

            rectangles.append(img)

        return rectangles

    def __create_vertical_bars(self, vertical_bar_width):
        vertical_bars = []
        for i in range(self.args.images_in_each_class):
            this_bar_width = randint(
                vertical_bar_width[0], vertical_bar_width[1])

            indexes = []
            k = 0
            for j in range(self.args.image_dimension):
                if k < this_bar_width:
                    indexes.append(j)

                k += 1
                if k >= this_bar_width * 4:
                    k = 0

            img = np.zeros((self.args.image_dimension,
                            self.args.image_dimension))

            img[indexes] = 1
            img = np.roll(img, randint(0, self.args.image_dimension), axis=0)

            vertical_bars.append(img)

        return vertical_bars

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
    # data, targets = data_generator.get_images()
    data_generator.random_sample()
