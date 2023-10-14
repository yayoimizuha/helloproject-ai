from typing import Any
from h5py import File
from io import BytesIO
from os import cpu_count
from os.path import join, pardir, dirname
from pandas import DataFrame
from sqlite3 import connect
from concurrent.futures import ProcessPoolExecutor as PPE
from multiprocessing import Manager
from sys import path, stdout, stderr
from shutil import move
from time import time

from tqdm import tqdm

path.append(pardir)
from settings import datadir

filename = join('/mnt/shm/blog_text.hdf5')
temporary_dir = '/mnt/shm/'
with open(file=filename, mode='rb') as f:
    file_bio = BytesIO(initial_bytes=f.read())


def extract(key: str, file: str) -> tuple[list[list[Any]], list[list[Any]]]:
    with File(name=file, mode='r') as hdf:
        print(f'start: {key}')
        start = time()
        article_table = []
        comment_table = []
        for blog_entry in hdf[key].keys():
            blog_article = hdf[key][blog_entry]['article'][()].decode('utf-8')
            entry_theme, entry_title, entry_date = list(hdf[key][blog_entry]['article'].attrs.values())
            article_table.append([blog_entry, key, entry_theme, entry_title, entry_date, blog_article])
            for comment_entry, comment_text in hdf[key][blog_entry]['comments_dataset'].items():
                comment_article = comment_text[()].decode('utf-8')
                comment_blog_id, comment_user_id, comment_nickname, comment_title, comment_date = \
                    list(comment_text.attrs.values())
                comment_table.append(
                    [comment_entry, comment_blog_id, comment_user_id, comment_nickname, comment_title, comment_date,
                     comment_article])
            # break
        print(f'end: {key} at {int(time() - start)}s')
        return article_table, comment_table


results = []
with PPE(max_workers=cpu_count()) as executor, File(name=file_bio, mode='r') as hdf5:
    for order, blog_group in enumerate(hdf5.keys(), start=0):
        # print(blog_group)
        lock = Manager().Lock()
        results.append(executor.submit(extract, blog_group, filename))

article_tables = []
comment_tables = []
for job in results:
    a, b = job.result()
    article_tables.extend(a)
    comment_tables.extend(b)

with connect(database=join(temporary_dir, 'tmp.sqlite'), timeout=3600) as connector:
    article_dataframe = DataFrame(data=article_tables, columns=['index', 'group', 'theme', 'title', 'date', 'article'])
    # article_dataframe.set_index('index')
    article_dataframe = article_dataframe.astype({"index":int})
    article_dataframe.to_sql(name='blog', con=connector, if_exists='replace',index=False)
    comment_dataframe = DataFrame(data=comment_tables,
                                  columns=['index', 'blog_id', 'user_id', 'nickname', 'title', 'date', 'article'])
    comment_dataframe.set_index('index')
    comment_dataframe.to_sql(name='comment', con=connector, if_exists='replace')

move(join(temporary_dir, 'tmp.sqlite'), join(datadir(), 'blog_post.sqlite'))
