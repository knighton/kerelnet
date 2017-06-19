from copy import deepcopy
import numpy as np
import os
from PIL import Image
import random
from time import time
import ujson as json


def load_questions(data_root, basename):
    print('Loading questions...')
    t0 = time()
    f = os.path.join(data_root, 'questions', basename)
    j = json.load(open(f))
    dd = j['questions']
    rr = []
    for d in dd:
        image = d['image_filename']
        question = d['question']
        answer = d.get('answer')
        rr.append((image, question, answer))
    t1 = time()
    print('Loading %s took %.3f sec.' % (f, t1 - t0))
    return rr


def load_data(data_root, split):
    basename = 'CLEVR_%s_questions.json' % split
    return load_questions(data_root, basename)
    

def load_image(f, shrink_ratio):
    im = Image.open(f)
    im = im.convert('RGB')
    new_size = map(lambda n: n // shrink_ratio, im.size)
    im = im.resize(new_size)
    im = np.array(im, dtype=np.float32)
    im /= 255
    return im


class CLEVR(object):
    def __init__(self, data_root, split, samples, batch_size=64,
                 image_shrink_ratio=4):
        self.data_root = data_root
        self.split = split
        self.samples = samples
        self.batch_size = batch_size
        self.image_shrink_ratio = image_shrink_ratio
        self.batches_per_epoch = len(self.samples) // self.batch_size
        self.batch = 0

    def __iter__(self):
        return self

    def load_image(self, name):
        f = os.path.join(self.data_root, 'images', self.split, name)
        return load_image(f, self.image_shrink_ratio)

    def get_image_shape(self):
        name = self.samples[0][0]
        return self.load_image(name).shape

    def __next__(self):
        if not self.batch:
            self.shuf_samples = deepcopy(self.samples)
            random.shuffle(self.shuf_samples)
        a = self.batch * self.batch_size
        z = (self.batch + 1) * self.batch_size
        samples = self.shuf_samples[a:z]
        names, questions, answers = zip(*samples)
        images = list(map(self.load_image, names))
        images = np.array(images, dtype=np.float32)
        questions = np.array(questions, dtype=np.int32)
        answers = np.array(answers, dtype=np.float32)
        self.batch += 1
        self.batch %= self.batches_per_epoch
        return [images, questions], answers
