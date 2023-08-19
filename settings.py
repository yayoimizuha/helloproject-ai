from functools import cache
from os import getcwd
from os.path import join

blog_list = ['angerme-ss-shin', 'angerme-amerika', 'angerme-new', 'juicejuice-official', 'tsubaki-factory',
             'morningmusume-9ki', 'morningmusume-10ki', 'mm-12ki', 'morningm-13ki', 'morningmusume15ki',
             'morningmusume16ki', 'beyooooonds-rfro', 'beyooooonds-chicatetsu', 'beyooooonds', 'ocha-norma',
             'countrygirls', 'risa-ogata',  # "shimizu--saki",
             'kumai-yurina-blog', 'sudou-maasa-blog', 'sugaya-risako-blog', 'miyamotokarin-official',
             'kobushi-factory', 'sayumimichishige-blog', 'kudo--haruka', 'airisuzuki-officialblog',
             'angerme-ayakawada', 'miyazaki-yuka-blog', 'tsugunaga-momoko-blog', 'tokunaga-chinami-blog',
             'c-ute-official', 'tanakareina-blog']


@cache
def theme_curator(theme: str, blog_id: str) -> str:
    if theme == "":
        theme = 'None'
    elif 'risa-ogata' == blog_id:
        theme = '小片リサ'
    elif 'shimizu--saki' == blog_id:
        theme = "清水佐紀"
    elif 'kumai-yurina-blog' == blog_id:
        theme = "熊井友理奈"
    elif 'sudou-maasa-blog' == blog_id:
        theme = "須藤茉麻"
    elif 'sugaya-risako-blog' == blog_id:
        theme = "菅谷梨沙子"
    elif 'miyamotokarin-official' == blog_id:
        theme = "宮本佳林"
    elif 'sayumimichishige-blog' == blog_id:
        theme = "道重さゆみ"
    elif 'kudo--haruka' == blog_id:
        theme = "工藤遥"
    elif 'airisuzuki-officialblog' == blog_id:
        theme = "鈴木愛理"
    elif 'angerme-ayakawada' == blog_id:
        theme = "和田彩花"
    elif 'miyazaki-yuka-blog' == blog_id:
        theme = "宮崎由加"
    elif 'tsugunaga-momoko-blog' == blog_id:
        theme = "嗣永桃子"
    elif 'natsuyaki-miyabi-blog' == blog_id:
        theme = "夏焼雅"
    elif 'tokunaga-chinami-blog' == blog_id:
        theme = "徳永千奈美"
    elif '梁川 奈々美' == theme:
        theme = '梁川奈々美'
    elif "tanakareina-blog" == blog_id:
        theme = "田中れいな"
    return theme


@cache
def datadir():
    return join(getcwd(), 'data')


request_header = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0'
}


class FaceCropProcesses:
    load = 1
    pre_process = 10
    predict = 3
    post_process = 4
