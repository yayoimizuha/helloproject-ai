from sqlite3 import connect
from shutil import copyfile
from settings import datadir
from os.path import join
from os import getcwd
from numpy import array, tile, int_, rot90, where, ndarray, vectorize, str_

# copyfile(src=join(datadir(), 'blog_post.sqlite'), dst=join('/mnt/shm/blog_post.sqlite'))
hash_func = vectorize(hash, otypes=[str])
with connect(database='/mnt/shm/blog_post.sqlite') as connector:
    cursor = connector.cursor()
    print(cursor.execute("SELECT t.sql FROM sqlite_master t WHERE name = 'blog'").fetchone())
    if 'article_cleaned' not in cursor.execute("SELECT t.sql FROM sqlite_master t WHERE name = 'blog'").fetchone()[0]:
        cursor.execute("ALTER TABLE blog add article_cleaned TEXT")
    print(cursor.execute("SELECT t.sql FROM sqlite_master t WHERE name = 'blog'").fetchone())

    for (theme,) in cursor.execute(f'SELECT DISTINCT theme FROM blog').fetchall():
        print(theme)
        blog_contents = cursor.execute(
            f'SELECT title,date,article,theme,"index" FROM blog WHERE theme = \'{theme}\' ORDER BY date').fetchall()
        # cleaned_list = [None] * (blog_contents.__len__() - 1)
        # cleaned_list = []
        for i in range(blog_contents.__len__() - 1):
            if blog_contents[i][3] != '八木栞':
                pass
            a = blog_contents[i]
            b = blog_contents[i + 1]
            # print(a)
            # print(b)
            a_list = array(a[2].split('\n'), dtype=object)
            b_list = array(b[2].split('\n'), dtype=object)
            a_hash_list = hash_func(a_list)
            b_hash_list = hash_func(b_list)
            # print(a_hash_list.shape)
            # print(b_hash_list.shape)
            a_ndarray: ndarray = tile(array(object=a_hash_list), reps=(*b_hash_list.shape, 1))
            b_ndarray: ndarray = rot90(tile(array(object=b_hash_list), reps=(*a_hash_list.shape, 1)))
            # print(a_ndarray.shape)
            # print(b_ndarray.shape)
            dup = a_ndarray == b_ndarray
            # print(dup)
            # print(list(zip(*where(dup))))
            cleaned_text = '\n'.join(a_list[(~dup.any(axis=0))]).replace('\'', '\'\'')
            cursor.execute(
                f"UPDATE blog SET article_cleaned = \'{cleaned_text}\' WHERE \"index\" = {blog_contents[i][4]}")
            # cleaned_list[i] = cleaned_text
            # cleaned_list.append(cleaned_text)
            print('\t' + blog_contents[i][0])
            # input()

    connector.cursor().close()
    connector.commit()
