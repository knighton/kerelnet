from kerelnet.pipe.base import Pipe


class Pipeline(Pipe):
    def __init__(self, steps):
        self.steps = steps

    def fit(self, x):
        for step in self.steps:
            x = step.fit_transform(x)

    def transform(self, x):
        for step in self.steps:
            x = step.transform(x)
        return x

    def inverse_transform(self, x):
        for step in reversed(self.steps):
            x = step.inverse_transform(x)
        return x
