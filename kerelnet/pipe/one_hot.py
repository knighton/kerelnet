from kerelnet.pipe.base import Pipe


class OneHot(Pipe):
    def fit(self, nn):
        self.vocab_size = max(nn) + 1

    def transform(self, nn):
        rrr = []
        for n in nn:
            rr = [0] * self.vocab_size
            rr[n] = 1
            rrr.append(rr)
        return rrr

    def inverse_transform(fff):
        return np.argmax(fff, axis=1)
