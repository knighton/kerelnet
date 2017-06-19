from collections import defaultdict
from glob import glob
import numpy as np
import os
from PIL import Image
import random
from time import time
from tqdm import tqdm
import ujson as json


def load_image(f):
    im = Image.open(f)
    im = im.convert('RGB')
    im = im.resize((200, 50))
    im = np.array(im, dtype=np.float32)
    im /= 255
    return im


def load_data(root_dir, split):
    assert split in {'train', 'dev', 'test'}

    samples = []
    f = os.path.join(root_dir, split, '%s.json' % split)
    for line in open(f):
        j = json.loads(line)
        sentence = j['sentence']
        label = j['label']
        name = j['identifier']
        samples.append((name, sentence, label))

    pattern = os.path.join(root_dir, split, 'images', '*', '*')
    ff = glob(pattern)

    name2ff = defaultdict(list)
    for f in ff:
        a = f.rfind(split) + len(split) + 1
        z = f.rfind('-')
        name = f[a:z]
        name2ff[name].append(f)

    print('Loading images...')
    rr = []
    t0 = time()
    i = 0
    for name, sentence, label in tqdm(samples):
        for f in name2ff[name]:
            image = load_image(f)
            rr.append((image, sentence, label))
            i += 1
    t = time() - t0
    print('Took %.3f sec.' % t)

    random.shuffle(rr)

    return rr
