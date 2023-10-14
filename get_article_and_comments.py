import sys
import re
import time
from io import BytesIO
from h5py import File, string_dtype
import requests
from numpy import array, ceil
from tqdm import tqdm
from settings import datadir, theme_curator
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from os.path import join
from bs4 import BeautifulSoup
import ujson
from more_itertools import chunked
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9), "JST")


def parse_article(url: str) -> tuple[str, str, str, str, str]:
    while True:
        with requests.get(url) as resp:
            html = resp.text
        try:
            json_obj = list(ujson.loads(re.findall(r'<script>window.INIT_DATA=(.*?)};', html)[0] + '}')['entryState'][
                                'entryMap'].values())[0]
            break
        except IndexError as e:
            print(e, url)
    blog_account = url.split('/')[-2]
    theme = theme_curator(json_obj['theme_name'], blog_account)
    date = json_obj['last_edit_datetime']
    blog_entry = json_obj['entry_id']
    try:
        entry_title = json_obj['entry_title']
    except:
        entry_title = ''
    entry_body = BeautifulSoup(json_obj['entry_text'].replace('<br>', '\n'), 'lxml')
    # print(entry_body)
    for emoji in entry_body.find_all('img', class_='emoji'):
        emoji.decompose()
    image_divs = entry_body.find_all('img', class_='PhotoSwipeImage')
    for div in image_divs:
        # print(div)
        if not div.has_attr('data-src'):
            entry_body.find('img', class_='PhotoSwipeImage').replaceWith(
                '\n' + '--blog-image-' + str(div["data-image-order"]) + '--')
    for i in entry_body.find_all('br'):
        i.replaceWith('\n')
    data_path = '/'.join([blog_account, str(blog_entry)])
    return entry_body.text, entry_title, theme, date, data_path


def get_api_json(api_url: str) -> list:
    while True:
        try:
            with requests.get(api_url) as resp:
                resp_json = ujson.loads(resp.text)
                comments_count = resp_json['paging']['total_count']
                break
        except Exception as e:
            time.sleep(5.0)
            print(api_url)
            print(e, resp.text, resp.status_code, file=sys.stderr)
    while True:
        if comments_count == 0:
            comments = []
            break
        else:
            try:
                with requests.get(api_url.replace('limit=1', f'limit={comments_count}')) as resp:
                    comments = list(ujson.loads(resp.text)['commentMap'].values())
                    break
            except Exception as e:
                time.sleep(5.0)
                print(e, file=sys.stderr)
    # print(comments.__len__())
    return comments


if __name__ == '__main__':
    chunk_size = 10
    article_executor = ProcessPoolExecutor(max_workers=cpu_count() * 2)
    api_executor = ProcessPoolExecutor(max_workers=chunk_size)

    hdf5_bio = BytesIO()
    with open(file=join(datadir(), 'blog_text.hdf5'), mode='rb') as hdf5_file:
        hdf5_bio.write(hdf5_file.read())

    save_cycle = 0
    num_lines = sum([1 for _ in open(file=join(datadir(), 'api_urls.txt'), mode='r')])
    with File(name=hdf5_bio, mode='a') as hdf5:
        with open(file=join(datadir(), 'api_urls.txt'), mode='r') as f:
            for rows in tqdm(chunked(f, n=chunk_size), total=ceil(num_lines / chunk_size)):
                # save_cycle += 1
                article_output = []
                api_output = []
                for row in rows:
                    article_url, comment_api_url = row.split(',')
                    blog_key = comment_api_url.split(';')[1].split('=')[1]
                    article_key = comment_api_url.split(';')[3].split('=')[1]
                    if f"/{blog_key}/{article_key}" in hdf5:
                        upd_time = datetime.fromisoformat(hdf5[blog_key][article_key]['article'].attrs['update_time'])
                        if (datetime.now(tz=JST) - upd_time).days > 4:
                            continue
                        else:
                            del hdf5[blog_key][article_key]
                    save_cycle += 1
                    article_output.append(article_executor.submit(parse_article, article_url))
                    api_output.append(api_executor.submit(get_api_json, comment_api_url))
                for article_res, api_res in zip(article_output, api_output):
                    entry_text, entry_title, theme, date, data_path = article_res.result()
                    comments = api_res.result()
                    post = hdf5.create_group(name=data_path)
                    article = post.create_dataset('article', dtype=string_dtype(encoding='utf-8'),
                                                  data=array(entry_text.encode('utf-8')))
                    article.attrs['theme'] = theme
                    article.attrs['title'] = entry_title
                    article.attrs['update_time'] = date

                    comments_dataset = post.create_group(name='comments_dataset')
                    if comments.__len__() != 0:
                        for order, text in enumerate(comments):
                            comment_id = str(text['comment_id'])
                            comment = comments_dataset.create_dataset(name=comment_id,
                                                                      dtype=string_dtype(encoding='utf-8'), data=array(
                                    text['comment_text'].replace('<br />', '\n').encode('utf-8')))
                            if 'comment_author' in text.keys():
                                comment.attrs['author_id'] = text['comment_author']['ameba_id']
                                comment.attrs['author_blog_id'] = text['comment_author']['blog_id']
                                comment.attrs['author_nickname'] = text['comment_author']['nickname']
                            else:
                                comment.attrs['author_id'] = ''
                                comment.attrs['author_blog_id'] = -1
                                comment.attrs['author_nickname'] = text['comment_name']
                            comment.attrs['comment_title'] = text['comment_title']
                            comment.attrs['comment_update_time'] = text['upd_datetime']
                hdf5.flush()
                if save_cycle > 1_000:
                    with open(file=join(datadir(), 'blog_text.hdf5'), mode='wb') as hdf5_file:
                        hdf5_file.write(hdf5_bio.getvalue())
                    save_cycle = 0
                    # exit()
    with open(file=join(datadir(), 'blog_text.hdf5'), mode='wb') as hdf5_file:
        hdf5_file.write(hdf5_bio.getvalue())
    article_executor.shutdown()
    api_executor.shutdown()
