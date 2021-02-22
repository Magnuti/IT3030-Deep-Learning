from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import math
from os import listdir

from data_generator_parser import DataGeneratorArguments


class Classes(Enum):
    ONE_RIGHT = 0
    TWO_RIGHT = 1
    ONE_LEFT = 2
    TWO_LEFT = 3


class DataGenerator:
    def __init__(self, args):
        self.args = args

        self.output_dir = pathlib.Path("sequences")
        self.output_dir.mkdir(exist_ok=True)

    def get_sequences(self):
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

    def generate_sequences(self):
        one_rights = self.__create_sequences(1)
        two_rights = self.__create_sequences(2)
        one_lefts = self.__create_sequences(-1)
        two_lefts = self.__create_sequences(-2)

        # Add noise
        one_rights = self.__add_noise(one_rights)
        two_rights = self.__add_noise(two_rights)
        one_lefts = self.__add_noise(one_lefts)
        two_lefts = self.__add_noise(two_lefts)

        def save_figures(figures, figure_name):
            for i, fig in enumerate(figures):
                fig *= 255
                plt.imsave(self.output_dir.joinpath(
                    "{}_{}.png".format(figure_name, i)), fig, cmap="gray")

        # Save sequences
        save_figures(one_rights, Classes.ONE_RIGHT.name.lower())
        save_figures(two_rights, Classes.TWO_RIGHT.name.lower())
        save_figures(one_lefts, Classes.ONE_LEFT.name.lower())
        save_figures(two_lefts, Classes.TWO_LEFT.name.lower())

    def __add_noise(self, sequences):
        for i, sequence in enumerate(sequences):
            for k in range(round(self.args.pattern_length * self.args.sequence_length * self.args.noise_ratio)):
                x = np.random.randint(0, self.args.pattern_length)
                y = np.random.randint(0, self.args.sequence_length)
                sequences[i][y, x] = not sequence[y, x]

        return sequences

    def __create_sequences(self, shift, threshold=0.7):
        sequences = []
        for _ in range(self.args.sequences):
            sequence = np.empty(
                (self.args.sequence_length, self.args.pattern_length))

            start_sequence = np.random.rand(self.args.pattern_length)

            # Adjust the number of 0s in the sequence.
            # If threshold == 0.5 we get an even distribution of 1s and 0s
            start_sequence[start_sequence <= threshold] = 0
            start_sequence[start_sequence > threshold] = 1

            # We do not want entirely black or white sequences
            while np.count_nonzero(start_sequence) == 0 or np.count_nonzero(start_sequence) == self.args.pattern_length:
                start_sequence = np.random.rand(self.args.pattern_length)

                start_sequence[start_sequence <= threshold] = 0
                start_sequence[start_sequence > threshold] = 1

            sequence[0] = start_sequence

            for i in range(1, self.args.sequence_length):
                sequence[i] = np.roll(sequence[i - 1], shift)

            sequences.append(sequence)
        return sequences


if __name__ == "__main__":
    args = DataGeneratorArguments()
    args.parse_arguments()
    data_generator = DataGenerator(args)
    data_generator.generate_sequences()
    # data, targets = data_generator.get_sequences()
    data_generator.random_sample()
