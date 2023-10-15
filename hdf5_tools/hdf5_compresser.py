import sys
from os.path import join, basename, dirname
from pprint import pprint
from io import BytesIO
from h5py import File
from datetime import datetime, timezone, timedelta
from gzip import compress, decompress

from tqdm import tqdm

JST = timezone(timedelta(hours=9), "JST")
COMPRESS_METHOD = 'gzip'
COMPRESS_OPT = 9
# print(sys.argv)
hdf5_bio = BytesIO()
hdf5_bio_compressed = BytesIO()
with open(file=join(sys.argv[1]), mode='rb') as hdf5_file:
    hdf5_bio.write(hdf5_file.read())

with File(name=hdf5_bio, mode='r') as hdf5, File(name=hdf5_bio_compressed, mode='w') as hdf5_compressed:
    for group in hdf5.keys():
        print(group)
        hdf5_group = hdf5_compressed.create_group(name=group)
        for article_id in tqdm(hdf5[group].keys()):
            article = hdf5_group.create_group(name=article_id)
            # print(group, article_id)
            article_txt = hdf5[group][article_id]['article']
            article_txt_compressed = article.create_dataset(name='article', dtype=f'S{article_txt[()].__len__() + 1}',
                                                            shape=(1,))
            article_txt_compressed[0] = article_txt[()]
            for k, v in article_txt.attrs.items():
                # print(k, v)
                article['article'].attrs[k] = v
            comments = article.create_group(name='comments_dataset')
            for comment_key in hdf5[group][article_id]['comments_dataset']:
                comment_txt = hdf5[group][article_id]['comments_dataset'][comment_key]
                # print(group, article_id, comment_key, comment_txt[()].decode('utf-8'))
                comment_txt_compressed = comments.create_dataset(name=comment_key,
                                                                 dtype=f'S{comment_txt[()].__len__() + 1}', shape=(1,))
                comment_txt_compressed[0] = comment_txt[()]
                for k, v in comment_txt.attrs.items():
                    comments[comment_key].attrs[k] = v

name, ext = basename(sys.argv[1]).rsplit('.', maxsplit=1)
with open(file=join(dirname(sys.argv[1]), name + '_compressed' + '.' + ext), mode='wb') as f:
    f.write(hdf5_bio_compressed.getvalue())

# None (bytes) 12.4 MiB (12,966,690)
# only article compressed 12.4 MiB (12,973,914)
# all gzipped 33.2 MiB (34,768,669)
# all chunked 33.2 MiB (34,768,669)
