import numpy as np
from data2d import sign


class LinReg:
    def __init__(self, dim) -> None:
        self.dim = dim
        self.w = np.array(dim + 1)

    def predict(self, x):
        n = x.shape[0]
        X = np.concatenate((np.ones((n, 1)), x), axis=1)
        raw_preds = X @ self.w
        return np.vectorize(sign)(raw_preds)

    def train(self, x, y):
        n = x.shape[0]
        X = np.concatenate((np.ones((n, 1)), x), axis=1)
        p_inv = np.linalg.inv(X.T @ X) @ X.T
        self.w = p_inv @ y

    def eval(self, x, y):
        preds = self.predict(x)
        return np.sum(preds == y) / y.shape[0]
