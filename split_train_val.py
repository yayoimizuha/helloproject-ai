from os import makedirs, listdir
from os.path import join
from settings import datadir
from shutil import rmtree, copyfile
from random import random
from tqdm import tqdm
from collections import OrderedDict
from asyncio import to_thread, gather, run
from aiofiles import open as a_open

valid_rate = 0.1
SRC_DIR = join(r'/mnt/share/dataset/vggface2/train')
DEST_DIR = join(datadir(), 'vggface2')

makedirs(DEST_DIR, exist_ok=True)
rmtree(join(DEST_DIR, 'train'), ignore_errors=True)
rmtree(join(DEST_DIR, 'val'), ignore_errors=True)
makedirs(join(DEST_DIR, 'train'), exist_ok=True)
makedirs(join(DEST_DIR, 'val'), exist_ok=True)


async def waiting(arg):
    return await gather(*arg)


async def async_copyfile(src_path: str, dst_path: str):
    async with a_open(file=src_path, mode='rb') as f:
        cont = await f.read()
    async with a_open(file=dst_path, mode='wb') as f:
        await f.write(cont)


with tqdm(listdir(SRC_DIR)) as pbar:
    for name in pbar:
        pbar.set_postfix(OrderedDict(name=name))
        # print(name)
        makedirs(join(DEST_DIR, 'train', name))
        makedirs(join(DEST_DIR, 'val', name))
        coroutines = []
        for file in listdir(join(SRC_DIR, name)):
            if random() > valid_rate:
                # copyfile(src=join(SRC_DIR, name, file),
                #          dst=join(DEST_DIR, 'train', name, file))
                # co = to_thread(copyfile, join(SRC_DIR, name, file), join(DEST_DIR, 'train', name, file))
                co = async_copyfile(join(SRC_DIR, name, file), join(DEST_DIR, 'train', name, file))
            else:
                # copyfile(src=join(SRC_DIR, name, file),
                #          dst=join(DEST_DIR, 'val', name, file))
                # co = to_thread(copyfile, join(SRC_DIR, name, file), join(DEST_DIR, 'val', name, file))
                co = async_copyfile(join(SRC_DIR, name, file), join(DEST_DIR, 'val', name, file))
            coroutines.append(co)
        # print(name, file)
        run(waiting(coroutines))

