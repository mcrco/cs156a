import numpy as np


def sign(x):
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


def rand_point():
    return np.random.uniform(-1, 1, 2)


def generate_line():
    p1 = rand_point()
    p2 = rand_point()
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return lambda x: m * x + b


def generate_data(n, line=None):
    line = line if line else generate_line()
    x = np.random.uniform(-1, 1, (n, 2))
    y = np.array([sign(xi[1] - line(xi[0])) for xi in x])  # .reshape((n, 1))
    return x, y


def load_data(file):
    with open(file, "r") as f:
        lines = f.readlines()
        floats = [[float(val) for val in line.split()] for line in lines]
        return np.array(floats)


def split_val(data, prop):
    np.random.shuffle(data)
    idx = round(prop * data.shape[0])
    return data[:idx], data[idx:]
