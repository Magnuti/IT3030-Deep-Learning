from enum import Enum


class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"
    SOFTMAX = "softmax"


class LossFunction(Enum):
    MSE = "MSE"
    CROSS_ENTROPY = "cross_entropy"


class GlobalWeightRegularizationOption(Enum):
    L1 = 0
    L2 = 1
