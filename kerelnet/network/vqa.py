from keras import backend as K
from keras.layers import *
from keras.models import Model
import numpy as np
from time import time


def see(x, initial_num_filters):
    i = 1
    while True:
        _, height, width, _ = K.int_shape(x)
        if height * width < 64:
            break
        x = Conv2D(initial_num_filters * i, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool2D((2, 2), 2)(x)
        i += 1
    x = Conv2D(initial_num_filters * i, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def read(x, vocab_size, token_embed_dim, lstm_dim):
    x = Embedding(vocab_size, token_embed_dim)(x)
    x = Bidirectional(LSTM(lstm_dim, implementation=2))(x)
    return x


def stack_layers(layers):
    def make_stack(x):
        for layer in layers:
            x = layer(x)
        return x
    return make_stack


def make_relater(num_stacks, dim):
    stack = []
    for i in range(num_stacks):
        stack += [
            Dense(dim),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5)
        ]
    return stack_layers(stack)


def make_classifier(x, num_stacks, dim, num_classes):
    for i in range(num_stacks):
        x = Dense(dim)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
    if num_classes < 2:
        assert False
    elif num_classes == 2:
        x = Dense(1, activation='sigmoid')(x)
    else:
        x = Dense(num_classes, activation='softmax')(x)
    return x


def crossentropy(num_classes):
    if num_classes < 2:
        assert False
    elif num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'
    return loss


def shrink_grid(x):
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D((2, 2), 2)(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def make_dense_vqa(
        input_image_shape, question_len, vocab_size, num_classes,
        initial_num_filters=32, token_embed_dim=64, lstm_dim=128,
        num_classifier_stacks=1, optim='adam'):
    image_input = Input(input_image_shape, dtype=np.float32)
    grid = see(image_input, initial_num_filters)
    grid = shrink_grid(grid)
    print('See() output: %s' % (K.int_shape(grid),))

    question_input = Input((question_len,), dtype=np.int32)
    question_embed = read(question_input, vocab_size, token_embed_dim, lstm_dim)
    print('Read() output: %s' % (K.int_shape(question_embed),))

    image_embed = Flatten()(grid)
    x = Concatenate()([image_embed, question_embed])
    print('Concatenate() output: %s' % (K.int_shape(x),))

    dim = K.int_shape(x)[1]
    x = make_classifier(x, num_classifier_stacks, dim, num_classes)
    print('Classifier output: %s' % (K.int_shape(x),))

    model = Model([image_input, question_input], x)
    model.summary()

    t0 = time()
    model.compile(optimizer=optim, loss=crossentropy(num_classes),
                  metrics=['accuracy'])
    t = time() - t0
    print('Model compilation took %.3f sec' % t)

    return model


def make_relational_vqa(
        input_image_shape, question_len, vocab_size, num_classes,
        initial_num_filters=32, token_embed_dim=64, lstm_dim=128,
        num_relater_stacks=2, relater_dim=256, num_classifier_stacks=3,
        optim='adam'):
    image_input = Input(input_image_shape, dtype=np.float32)
    grid = see(image_input, initial_num_filters)
    print('See() output: %s' % (K.int_shape(grid),))

    question_input = Input((question_len,), dtype=np.int32)
    question_embed = read(question_input, vocab_size, token_embed_dim, lstm_dim)
    print('Read() output: %s' % (K.int_shape(question_embed),))

    relater = make_relater(num_relater_stacks, relater_dim)
    t0 = time()
    x = Relational2D(relater, question_embed)(grid)
    t = time() - t0
    print('Constructing Relational2D() took %.3f sec' % t)
    print('Relational2D() output: %s' % (K.int_shape(x),))

    x = make_classifier(x, num_classifier_stacks, relater_dim, num_classes)
    print('Classifier output: %s' % (K.int_shape(x),))

    model = Model([image_input, question_input], x)
    model.summary()

    t0 = time()
    model.compile(optimizer=optim, loss=crossentropy(num_classes),
                  metrics=['accuracy'])
    t = time() - t0
    print('Model compilation took %.3f sec' % t)

    return model
