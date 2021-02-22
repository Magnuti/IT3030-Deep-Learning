from enum import Enum


class ActivationFunction(Enum):
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    LINEAR = "linear"
    SOFTMAX = "softmax"


class LayerType(Enum):
    DENSE = "dense"
    RECURRENT = "recurrent"


class LossFunction(Enum):
    MSE = "MSE"
    CROSS_ENTROPY = "cross_entropy"
