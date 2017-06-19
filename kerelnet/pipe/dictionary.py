from kerelnet.pipe.base import Pipe


class Dictionary(Pipe):
    def __init__(self):
        self.s2n = {}
        self.n2s = {}

    def vocab_size(self):
        return len(self.s2n)

    def add_token(self, token):
        n = self.s2n.get(token)
        if n is not None:
            return
        n = len(self.s2n)
        self.s2n[token] = n
        self.n2s[n] = token

    def fit(self, tokens):
        for token in tokens:
            self.add_token(token)

    def transform(self, tokens):
        return list(map(lambda token: self.s2n[token], tokens))

    def inverse_transform(self, nn):
        return list(map(lambda n: self.n2s[n], nn))
