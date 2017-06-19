import os

from kerelnet.dataset.clevr import CLEVR, load_data
from kerelnet.network.vqa import make_dense_vqa, make_relational_vqa
from kerelnet.pipe import *


def main():
    data_dir = os.path.join('data', 'CLEVR_v1.0')
    num_epochs = 100
    batch_size = 64
    image_shrink_ratio = 4
    network = 'dense_vqa'

    train_data = load_data(data_dir, 'train')
    names, questions, answers = zip(*train_data)

    question_pipe = Pipeline([
        Tokenizer(),
        MaxLenPadder(),
        Numpy(),
    ])
    questions = question_pipe.fit_transform(questions)
    print('Questions: %s' % (questions.shape,))

    answer_pipe = Pipeline([
        Dictionary(),
        OneHot(),
        Numpy(),
    ])
    answers = answer_pipe.fit_transform(answers)
    num_classes = answer_pipe.steps[0].vocab_size()
    print('Answers: %s' % (answers.shape,))

    samples = list(zip(names, questions, answers))
    train_data = CLEVR(data_dir, 'train', samples, batch_size,
                       image_shrink_ratio)
    input_image_shape = train_data.get_image_shape()
    print('Each image: %s' % (input_image_shape,))
    question_len = questions.shape[1]
    vocab_size = question_pipe.steps[0].vocab_size()
    print('Vocab size: %d' % vocab_size)

    if network == 'dense_vqa':
        initial_num_filters = 32
        token_embed_dim = 64
        lstm_dim = 128
        num_classifier_stacks = 1
        optim = 'adam'
        model = make_dense_vqa(
            input_image_shape, question_len, vocab_size, num_classes,
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
            input_image_shape, question_len, vocab_size, num_classes,
            initial_num_filters, token_embed_dim, lstm_dim, num_relater_stacks,
            relater_dim, num_classifier_stacks, optim)
    else:
        assert False

    val_data = load_data(data_dir, 'val')
    names, questions, answers = zip(*val_data)
    questions = question_pipe.transform(questions)
    answers = answer_pipe.transform(answers)
    samples = list(zip(names, questions, answers))
    val_data = CLEVR(data_dir, 'val', samples, batch_size, image_shrink_ratio)

    train_batches_per_epoch = len(samples) // batch_size // 10
    val_batches_per_epoch = train_batches_per_epoch // 5
    model.fit_generator(
        train_data, train_batches_per_epoch, epochs=num_epochs,
        validation_data=val_data, validation_steps=val_batches_per_epoch)


if __name__ == '__main__':
    main()
