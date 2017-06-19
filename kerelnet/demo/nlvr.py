from keras import backend as K
from keras.layers import *
from keras.models import Model
import numpy as np
import os
from time import time

from kerelnet.dataset.nlvr import load_data
from kerelnet.pipe import *
from kerelnet.network.vqa import make_dense_vqa, make_relational_vqa


def main():
    data_dir = os.path.join('data', 'nlvr')
    num_epochs = 100
    batch_size = 64
    network = 'dense_vqa'

    train_data = load_data(data_dir, 'train')
    images, sentences, labels = zip(*train_data)

    images = np.array(images, dtype=np.float32)
    print('Images: %s' % (images.shape,))

    sentence_pipe = Pipeline([
        Tokenizer(),
        MaxLenPadder(),
        Numpy(),
    ])
    sentences = sentence_pipe.fit_transform(sentences)
    print('Sentences: %s' % (sentences.shape,))

    label_pipe = Pipeline([
        TrueFalse(),
        Numpy(),
    ])
    labels = label_pipe.fit_transform(labels)
    num_classes = 2
    print('Labels: %s' % (labels.shape,))

    input_image_shape = tuple(images.shape[1:])
    sentence_len = sentences.shape[1]
    vocab_size = sentence_pipe.steps[0].vocab_size()
    print('Vocab size: %d' % vocab_size)

    if network == 'dense_vqa':
        initial_num_filters = 32
        token_embed_dim = 64
        lstm_dim = 128
        num_classifier_stacks = 1
        optim = 'adam'
        model = make_dense_vqa(
            input_image_shape, sentence_len, vocab_size, num_classes,
            initial_num_filters, token_embed_dim, lstm_dim,
            num_classifier_stacks, optim)
    elif network == 'relational_vqa':
        initial_num_filters = 32
        token_embed_dim = 64
        lstm_dim = 128
        num_relater_stacks = 2
        relater_dim = 256
        num_classifier_stacks = 3
        optim = 'adam'
        model = make_relational_vqa(
            input_image_shape, sentence_len, vocab_size, num_classes,
            initial_num_filters, token_embed_dim, lstm_dim, num_relater_stacks,
            relater_dim, num_classifier_stacks, optim)
    else:
        assert False

    dev_data = load_data(data_dir, 'dev')
    dev_images, dev_sentences, dev_labels = zip(*dev_data)
    dev_images = np.array(dev_images, dtype=np.float32)
    dev_sentences = sentence_pipe.transform(dev_sentences)
    dev_labels = label_pipe.transform(dev_labels)
    dev_data = [dev_images, dev_sentences], dev_labels

    model.fit([images, sentences], labels, batch_size=batch_size,
              epochs=num_epochs, validation_data=dev_data)


if __name__ == '__main__':
    main()
