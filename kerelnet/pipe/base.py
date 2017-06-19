DNE = 0
UNK = 1


class Pipe(object):
    def fit(self, x):
        raise NotImplementedError

    def transform(self, x):
        raise NotImplementedError

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        raise NotImplementedError
