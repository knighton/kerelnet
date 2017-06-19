from keras import backend as K
from keras.layers import Add, Concatenate, Lambda


def _relate_combinations(relater_network, features, concat_to_each_pair=None):
    if concat_to_each_pair is not None:
        assert K.ndim(concat_to_each_pair) == 2

    pairs = []
    if concat_to_each_pair is None:
        for a in features:
            for b in features:
                pairs.append(Concatenate()([a, b]))
    else:
        for a in features:
            for b in features:
                pairs.append(Concatenate()([a, b, concat_to_each_pair]))

    outputs = list(map(lambda pair: relater_network(pair), pairs))
    return Add()(outputs)


def Relational1D(relater_network, concat_to_each_pair=None):
    def f(seq):
        assert K.ndim(seq) == 3

        features = []
        length = K.int_shape(seq)[1]
        for i in range(length):
            features.append(Lambda(lambda x: x[:, i, :])(seq))

        return _relate_combinations(relater_network, features,
                                    concat_to_each_pair)

    return Lambda(f)


def Relational2D(relater_network, concat_to_each_pair=None):
    def f(grid):
        assert K.ndim(grid) == 4

        features = []
        _, height, width, _ = K.int_shape(grid)
        for i in range(height):
            for j in range(width):
                features.append(Lambda(lambda x: x[:, i, j, :])(grid))

        return _relate_combinations(relater_network, features,
                                    concat_to_each_pair)

    return Lambda(f)
