from pprint import pprint
from typing import List, Tuple

import h5py

import settings
import re
import sys
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from aiohttp import ClientSession, ClientConnectorError, ClientTimeout
from itertools import chain
from asyncio import run, Semaphore, sleep, Lock
from datetime import datetime
from aiofiles import open as a_open
import time
from os import path, utime, stat, cpu_count, makedirs
from tqdm.asyncio import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor, Future, ThreadPoolExecutor
from ujson import loads
from warnings import filterwarnings
from h5py import File, special_dtype, string_dtype
from io import BytesIO
from numpy import void, array
import ujson
import orjson
import requests

PARALLEL_LIMIT = 300

makedirs(path.join(settings.datadir(), 'blog_images'), exist_ok=True)
makedirs(path.join(settings.datadir(), 'blog_text'), exist_ok=True)

filterwarnings('ignore', category=MarkupResemblesLocatorWarning, module='bs4')


async def run_each(name: str) -> None:
    sem: Semaphore = Semaphore(PARALLEL_LIMIT)
    session: ClientSession = ClientSession(trust_env=True, headers=settings.request_header,
                                           timeout=ClientTimeout(total=10 * 60))
    list_pages_count = await parse_list_pages_count(name)

    print(name, list_pages_count)

    url_lists = await tqdm.gather(*[parse_list_page(name, i, sem, session) for i in range(1, list_pages_count + 1)],
                                  desc=name)

    url_list = list(chain.from_iterable(url_lists))

    for url in url_list:
        if 'html' not in url:
            print(url)
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    lock = Lock()
    futures = await tqdm.gather(
            *[parse_blog_post(url, sem, session, executor, lock) for url in url_list],
            desc='scan blog')
    executor.shutdown()
    image_link_package = list(chain.from_iterable(futures))

    await tqdm.gather(
        *[download_image(filename, url, date, sem, session) for filename, url, date in image_link_package],
        desc='downloading images')

    await session.close()


async def parse_list_pages_count(blog_name: str) -> int:
    async with ClientSession(trust_env=True, headers=settings.request_header) as session:
        async with session.get(f'https://ameblo.jp/{blog_name}/entrylist.html') as resp:
            resp_html = await resp.text()
            json_obj = ujson.loads(re.findall(r'<script>window.INIT_DATA=(.*?)};', resp_html)[0] + '}')
            return list(json_obj['entryState']['blogPageMap'].values())[0]['paging']['max_page']


async def parse_list_page(blog_name: str, order: int, sem: Semaphore, session: ClientSession) -> list[str]:
    async with sem:
        async with session.get(f'https://ameblo.jp/{blog_name}/entrylist-{order}.html') as resp:
            resp_html = await resp.text()
    try:
        json_obj = ujson.loads(re.findall(r'<script>window.INIT_DATA=(.*?)};', resp_html)[0] + '}')
        page_url_list: list[str] = list()
        for blog_post_desc in list(json_obj['entryState']['entryMap'].values()):
            if blog_post_desc['publish_flg'] == 'open':
                page_url_list.append(f"https://ameblo.jp/{blog_name}/entry-{blog_post_desc['entry_id']}.html" +
                                     "," +
                                     ";".join(["https://ameblo.jp/_api/blogComments", f"amebaId={blog_name}",
                                               f"blogId={blog_post_desc['blog_id']}",
                                               f"entryId={blog_post_desc['entry_id']}",
                                               "excludeReplies=false", "limit=1", "offset=0"]))
    except Exception as e:
        print(e)
        print(f'https://ameblo.jp/{blog_name}/entrylist-{order}.html')
        return []
    return page_url_list


def parse_image(html: str, url: str) -> list[tuple[str, str, datetime]]:
    blog_account = url.split('/')[-2]
    try:
        json_obj = list(ujson.loads(re.findall(r'<script>window.INIT_DATA=(.*?)};', html)[0] + '}')['entryState'][
                            'entryMap'].values())[0]
    except IndexError as e:
        print(e, url)
        exit()
    theme = settings.theme_curator(json_obj['theme_name'], blog_account)
    date = datetime.fromisoformat(json_obj['last_edit_datetime'])
    blog_entry = json_obj['entry_id']
    entry_body = BeautifulSoup(json_obj['entry_text'].replace('<br>', '\n'), 'lxml')
    # print(entry_body)
    for emoji in entry_body.find_all('img', class_='emoji'):
        emoji.decompose()
    image_divs = entry_body.find_all('img', class_='PhotoSwipeImage')
    return_list = list()
    for div in image_divs:
        # print(div)
        if not div.has_attr('data-src'):
            return_list.append((
                '='.join([theme, blog_account, str(blog_entry)]) + '-' + str(div['data-image-order']) + '.jpg',
                str(div['src']).split('?')[0],
                date
            ))
            entry_body.find('img', class_='PhotoSwipeImage').replaceWith(
                '--blog-image-' + str(div["data-image-order"]) + '--\n')
    return return_list


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


async def parse_blog_post(urls: str, sem: Semaphore, session: ClientSession, executor: ProcessPoolExecutor,
                          lock: Lock) -> Future:
    page_url, comment_api_url = urls.split(',')
    async with sem:
        while True:
            try:
                async with session.get(page_url) as resp:
                    resp_html = await resp.text()
                    # await sleep(1.0)
                    break
            except ClientConnectorError as e:
                await sleep(5.0)
                print(e, file=sys.stderr)

    o = executor.submit(parse_image, resp_html, page_url)
    async with lock:
        async with a_open(file=path.join(settings.datadir(), 'api_urls.txt'), mode='a') as f:
            await f.write(urls + '\n')

    image_list = o.result()
    return image_list


async def download_image(filename: str, url: str, date: datetime, sem: Semaphore, session: ClientSession) -> None:
    tag = filename.split('=')[0]
    if not path.isdir(path.join(settings.datadir(), "blog_images", tag)):
        makedirs(path.join(settings.datadir(), "blog_images", tag), exist_ok=True)
    filepath = path.join(settings.datadir(), "blog_images", tag, filename)
    if path.isfile(filepath):
        # print(f"file already downloaded.: {filename}")
        return
    async with sem:
        # print("download: ", url)
        async with session.get(url) as resp:
            if resp.content_type != "image/jpeg":
                return
            async with a_open(file=filepath, mode="wb") as f:
                await f.write(await resp.read())
    utime(path=filepath, times=(stat(path=filepath).st_atime, date.timestamp()))


theme_regex = re.compile('"theme_name":"(.*?)"')
modified_time_regex = re.compile('"dateModified":"(.*?)"')


def grep_theme(html: str) -> str:
    return str(theme_regex.search(html).group(1))


def grep_modified_time(html: str) -> str:
    return str(modified_time_regex.search(html).group(1))


if __name__ == '__main__':
    with open(file=path.join(settings.datadir(),'api_urls.txt'),mode='w') as f:
        f.write("")
    for blog in settings.blog_list:
        run(run_each(blog))
