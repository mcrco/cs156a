from linreg import LinReg


class LinRegTransform:
    def __init__(self, transform, dim) -> None:
        self.transform = transform
        self.linreg = LinReg(dim)

    def predict(self, x):
        x = self.transform(x)
        return self.linreg.predict(x)

    def train(self, x, y):
        x = self.transform(x)
        self.linreg.train(x, y)

    def eval(self, x, y):
        x = self.transform(x)
        return self.linreg.eval(x, y)
