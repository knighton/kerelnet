from kerelnet.pipe.base import DNE, Pipe, UNK


class Tokenizer(Pipe):
    def __init__(self):
        self.s2n = {}
        self.n2s = {}

    def vocab_size(self):
        return len(self.s2n) + 2

    def tokenize(self, text):
        text = text.lower()
        cc = []
        for c in text:
            if 'a' <= c <= 'z':
                cc.append(c)
            elif '0' <= c <= '9':
                cc.append(c)
            else:
                cc.append(' ')
        return ''.join(cc).split()

    def add_token(self, s):
        n = self.s2n.get(s)
        if n is not None:
            return
        n = len(self.s2n) + 2
        self.s2n[s] = n
        self.n2s[n] = s

    def fit(self, texts):
        texts = map(lambda s: s.lower(), texts)
        for text in texts:
            for token in self.tokenize(text):
                self.add_token(token)

    def transform(self, texts):
        nnn = []
        for text in texts:
            nn = []
            for token in self.tokenize(text):
                n = self.s2n.get(token, UNK)
                nn.append(n)
            nnn.append(nn)
        return nnn
