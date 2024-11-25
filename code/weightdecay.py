import numpy as np
from linreg import LinReg

class LinRegWD(LinReg):
    def __init__(self, dim, decay_factor) -> None:
        self.dim = decay_factor
        self.decay_factor = decay_factor
        self.w = np.array(dim + 1)

    def train(self, x, y):
        n = x.shape[0]
        X = np.concatenate((np.ones((n, 1)), x), axis=1)
        self.w = np.linalg.pinv(
            X.T @ X + self.decay_factor * np.identity(X.shape[1])
        ) @ X.T @ y
