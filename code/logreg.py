import numpy as np
import math
import random

class LogReg():
    def __init__(self, dim, lr=0.01) -> None:
        self.dim = dim
        self.w = np.zeros(dim + 1)
        self.lr = lr
        self.epochs = 0

    def predict(self, x):
        x = np.concatenate((np.ones(1), x))
        s = self.w @ x
        exp = math.exp(s)
        return exp / (1 + exp)

    def step(self, x, y):
        x = np.concatenate((np.ones(1), x))
        grad = -y * x / (1 + math.exp(y * self.w @ x))
        self.w -= self.lr * grad
        return -self.lr * grad

    def sgd(self, X, Y):
        permutation = [i for i in range(X.shape[0])]
        random.shuffle(permutation)
        diff = np.zeros(self.dim + 1)
        for i in permutation:
            diff += self.step(X[i], Y[i])
        return diff

    def train(self, X, Y):
        w_diff = np.ones(self.dim + 1)
        while math.sqrt(w_diff @ w_diff) >= 0.01:
            w_diff = self.sgd(X, Y)
            self.epochs += 1

    def eval(self, X, Y):
        preds = np.array([self.predict(x) for x in X])
        for pred, y in zip(preds, Y):
            print(pred.item(), y.item())

    def error(self, X, Y):
        X = np.concat((np.ones((X.shape[0], 1)), X), axis=1)
        cross_entropy = lambda x, y: math.log(1 + math.exp(-y * self.w @ x))
        errors = [cross_entropy(x, y) for x, y in zip(X, Y)]
        return sum(errors) / len(errors)


def binary(x):
    if x < 0:
        return 0
    return 1


def rand_point():
    return np.random.uniform(-1, 1, 2)


def generate_line():
    p1 = rand_point()
    p2 = rand_point()
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    f = lambda x:  m * x + b
    return f


def generate_data(n, f=None):
    if f is None:
        line = generate_line()
        f = lambda x: binary(x[1] - line(x[0]))
    x = np.random.uniform(-1, 1, (n, 2))
    y = np.array([f(xi) for xi in x]).reshape((n, 1))

    return x, y


def p8p9p10():
    num_trials = 100
    num_points = 100
    epochs = []
    test_errors = []

    for _ in range(num_trials):
        line = generate_line()
        f = lambda x: binary(x[1] - line(x[0]))
        x, y = generate_data(num_points, f)
        model = LogReg(2, 0.01)
        model.train(x, y)
        # model.eval(x, y)
        epochs.append(model.epochs)

        x_test, y_test = generate_data(num_points, f)
        test_errors.append(model.error(x_test, y_test))

    print(f"Average test error: {sum(test_errors) / len(test_errors)}")
    print(f"Average epochs for training: {sum(epochs) / len(epochs)}")

if __name__ == "__main__":
    p8p9p10()
