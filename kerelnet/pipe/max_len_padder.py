from kerelnet.pipe.base import DNE, Pipe


class MaxLenPadder(Pipe):
    def __init__(self):
        self.to_len = None

    def fit(self, nnn):
        self.to_len = max(map(len, nnn))

    def transform(self, nnn):
        shape = len(nnn), self.to_len
        rrr = []
        for nn in nnn:
            if len(nn) < self.to_len:
                rr = nn + [DNE] * (self.to_len - len(nn))
            else:
                rr = nn[:self.to_len]
            rrr.append(rr)
        return rrr
