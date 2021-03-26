from enum import Enum


class Dataset(Enum):
    MNIST = "MNIST"
    FASHION_MNIST = "FASHION_MNIST"


class LossFunction(Enum):
    MSE = "MSE"
    CROSS_ENTROPY = "cross_entropy"


class Optimizer(Enum):
    SGD = "SGD"
    ADAM = "Adam"
