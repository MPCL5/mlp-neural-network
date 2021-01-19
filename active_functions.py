import math


def identity(x):
    return x


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def bipolar_sigmoid(x):
    return 2 / (1 + math.exp(-x)) - 1


def derivative_bipolar_sigmoid(x):
    return (1/2) * (1+bipolar_sigmoid(x)) * (1-bipolar_sigmoid(x))
