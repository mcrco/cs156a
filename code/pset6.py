from linregtrans import LinRegTransform
from weightdecay import LinRegWD
import numpy as numpy


class LinRegTransformWD:
    def __init__(self, transform, dim, decay) -> None:
        self.transform = transform
        self.linreg = LinRegWD(dim, decay)

    def predict(self, x):
        x = self.transform(x)
        return self.linreg.predict(x)

    def train(self, x, y):
        x = self.transform(x)
        self.linreg.train(x, y)

    def eval(self, x, y):
        x = self.transform(x)
        return self.linreg.eval(x, y)


def quadratic_transform(X):
    x1sq = (X[:, 0] ** 2).reshape(X.shape[0], 1)
    x2sq = (X[:, 1] ** 2).reshape(X.shape[0], 1)
    x1x2 = (X[:, 0] * X[:, 1]).reshape(X.shape[0], 1)
    x1subx2 = (np.abs(X[:, 0] - X[:, 1])).reshape(X.shape[0], 1)
    x1plusx2 = (np.abs(X[:, 0] + X[:, 1])).reshape(X.shape[0], 1)
    X = np.concatenate((X, x1sq, x2sq, x1x2, x1subx2, x1plusx2), axis=1)
    return X


def load_data(file):
    with open(file, "r") as f:
        lines = f.readlines()
        floats = [[float(val) for val in line.split()] for line in lines]
        return np.array(floats)


def p2():
    model = LinRegTransform(quadratic_transform, 6)

    in_data = load_data("../data/in.dta")
    x_in = in_data[:, :2]
    y_in = in_data[:, 2:]

    out_data = load_data("../data/out.dta")
    x_out = out_data[:, :2]
    y_out = out_data[:, 2:]

    model.train(x_in, y_in)
    in_acc = model.eval(x_in, y_in)
    out_acc = model.eval(x_out, y_out)
    print("Problem 2")
    print(f"Training error: {1 - in_acc}")
    print(f"Testing error: {1 - out_acc}\n")


def p3():
    model = LinRegTransformWD(quadratic_transform, 6, 1e-3)

    in_data = load_data("../data/in.dta")
    x_in = in_data[:, :2]
    y_in = in_data[:, 2:]

    out_data = load_data("../data/out.dta")
    x_out = out_data[:, :2]
    y_out = out_data[:, 2:]

    model.train(x_in, y_in)
    in_acc = model.eval(x_in, y_in)
    out_acc = model.eval(x_out, y_out)
    print("Problem 3")
    print(f"Training error: {1 - in_acc}")
    print(f"Testing error: {1 - out_acc}\n")


def p4():
    model = LinRegTransformWD(quadratic_transform, 6, 1e3)

    in_data = load_data("../data/in.dta")
    x_in = in_data[:, :2]
    y_in = in_data[:, 2:]

    out_data = load_data("../data/out.dta")
    x_out = out_data[:, :2]
    y_out = out_data[:, 2:]

    model.train(x_in, y_in)
    in_acc = model.eval(x_in, y_in)
    out_acc = model.eval(x_out, y_out)
    print("Problem 4")
    print(f"Training error: {1 - in_acc}")
    print(f"Testing error: {1 - out_acc}\n")


def p5():
    print("Problem 5")
    for k in [-2, -1, 0, 1, 2]:
        model = LinRegTransformWD(quadratic_transform, 6, 10**k)

        in_data = load_data("../data/in.dta")
        x_in = in_data[:, :2]
        y_in = in_data[:, 2:]

        out_data = load_data("../data/out.dta")
        x_out = out_data[:, :2]
        y_out = out_data[:, 2:]

        model.train(x_in, y_in)
        in_acc = model.eval(x_in, y_in)
        out_acc = model.eval(x_out, y_out)
        print(f"Training error with k = {k}: {1 - in_acc}")
        print(f"Testing error with k = {k}: {1 - out_acc}")
    print("")


def p6():
    best = 1e10
    for k in range(-10, 11):
        model = LinRegTransformWD(quadratic_transform, 6, 10**k)

        in_data = load_data("../data/in.dta")
        x_in = in_data[:, :2]
        y_in = in_data[:, 2:]

        out_data = load_data("../data/out.dta")
        x_out = out_data[:, :2]
        y_out = out_data[:, 2:]

        model.train(x_in, y_in)
        out_acc = model.eval(x_out, y_out)
        best = min(best, 1 - out_acc)
    print("Problem 6")
    print(f"Best out-of-sample error for k from -10 to 10: {best}")


if __name__ == "__main__":
    p2()
    # Problem 2
    # Training error: 0.02857142857142858
    # Testing error: 0.08399999999999996

    p3()
    # Problem 3
    # Training error: 0.02857142857142858
    # Testing error: 0.07999999999999996

    p4()
    # Problem 4
    # Training error: 0.37142857142857144
    # Testing error: 0.43600000000000005

    p5()
    # Problem 5
    # Training error with k = -2: 0.02857142857142858
    # Testing error with k = -2: 0.08399999999999996
    # Training error with k = -1: 0.02857142857142858
    # Testing error with k = -1: 0.05600000000000005
    # Training error with k = 0: 0.0
    # Testing error with k = 0: 0.09199999999999997
    # Training error with k = 1: 0.05714285714285716
    # Testing error with k = 1: 0.124
    # Training error with k = 2: 0.19999999999999996
    # Testing error with k = 2: 0.22799999999999998

    p6()
    # Problem 6
    # Best out-of-sample error for k from -10 to 10: 0.05600000000000005
