from h5py import File
from io import BytesIO
from os import getcwd
from os.path import join
from pandas import DataFrame, to_datetime, concat
from sqlite3 import connect
from tqdm import tqdm

filename = join('/mnt/shm/blog_text.hdf5')

with open(file=filename, mode='rb') as f:
    file_bio = BytesIO(initial_bytes=f.read())

article_tables = []
comment_tables = []
with (File(name=file_bio, mode='r+') as hdf5):
    for blog_group in hdf5.keys():
        print(blog_group)
        article_table = DataFrame(columns=['group', 'theme', 'title', 'date', 'article'])
        for blog_entry in tqdm(hdf5[blog_group].keys()):
            # print(blog_entry)
            # print(hdf5[blog_group][blog_entry].keys())
            blog_article = hdf5[blog_group][blog_entry]['article'][()].decode('utf-8')
            (_, entry_theme), (_, entry_title), (_, entry_date) = \
                list(hdf5[blog_group][blog_entry]['article'].attrs.items())
            # print(entry_theme, entry_title, entry_date)
            article_table.loc[blog_entry] = [blog_group, entry_theme, entry_title, entry_date, blog_article]
            comment_table = DataFrame(columns=['blog_id', 'user_id', 'nickname', 'title', 'date', 'article'])
            for comment_entry, comment_text in hdf5[blog_group][blog_entry]['comments_dataset'].items():
                comment_article = comment_text[()].decode('utf-8')
                comment_blog_id, comment_user_id, comment_nickname, comment_title, comment_date = \
                    list(comment_text.attrs.values())
                comment_table.loc[comment_entry] = \
                    [comment_blog_id, comment_user_id, comment_nickname, comment_title, comment_date, comment_article]
            comment_tables.append(comment_table.copy(deep=True))
        article_tables.append(article_table.copy(deep=True))

        # break

# print(table)
with connect('blog_post.sqlite') as connector:
    concat(objs=article_tables).to_sql(name='blog', con=connector)
    concat(comment_tables).to_sql(name='comments', con=connector)
