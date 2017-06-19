from kerelnet.pipe.base import Pipe


class TrueFalse(Pipe):
    def fit(self, tokens):
        pass

    def transform(self, tokens):
        return list(map(lambda s: int(bool(s)), tokens))

    def inverse_transform(ff):
        return list(map(lambda f: 'false' if f < 0.5 else 'true', ff))
