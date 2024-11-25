from sklearn.svm import SVC
import numpy as np


class SVM:
    def __init__(self) -> None:
        self.svc = SVC(kernel="linear", C=1e10)

    def train(self, x, y):
        self.svc.fit(x, y)

    def predict(self, x):
        return self.svc.predict(x)

    def eval(self, x, y):
        acc = np.sum(self.predict(x) == y) / x.shape[0]
        n_support = self.svc.n_support_
        return acc, n_support
