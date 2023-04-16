from os import makedirs, listdir
from os.path import join
from settings import datadir
from shutil import rmtree, copyfile
from random import random

valid_rate = 0.1

makedirs(join(datadir(), 'dataset'), exist_ok=True)
rmtree(join(datadir(), 'dataset', 'train'), ignore_errors=True)
rmtree(join(datadir(), 'dataset', 'val'), ignore_errors=True)
makedirs(join(datadir(), 'dataset', 'train'), exist_ok=True)
makedirs(join(datadir(), 'dataset', 'val'), exist_ok=True)

for name in listdir(join(datadir(), 'sample_set')):
    print(name)
    makedirs(join(datadir(), 'dataset', 'train', name))
    makedirs(join(datadir(), 'dataset', 'val', name))
    for file in listdir(join(datadir(), 'sample_set', name)):
        if random() > valid_rate:
            copyfile(src=join(datadir(), 'sample_set', name, file),
                     dst=join(datadir(), 'dataset', 'train', name, file))
        else:
            copyfile(src=join(datadir(), 'sample_set', name, file),
                     dst=join(datadir(), 'dataset', 'val', name, file))

    # print(name, file)
