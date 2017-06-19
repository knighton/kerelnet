import numpy as np

from kerelnet.pipe.base import Pipe


class Numpy(Pipe):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def fit(self, aa):
        pass

    def transform(self, aa):
        return np.array(aa, dtype=self.dtype)

    def inverse_transform(self, aa):
        return list(map(list, aa))
