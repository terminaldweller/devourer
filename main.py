#!/usr/bin/env python3
# _*_ coding=utf-8 _*_

import argparse
import logging
from newspaper import Article, build, Config
from bs4 import BeautifulSoup
from contextlib import closing
from requests import get
from requests.exceptions import RequestException
from re import findall


class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--source",
            type=str, help="the url where the urls to be extracted reside")
        parser.add_argument("--bool", action="store_true",
                            help="bool", default=False)
        self.args = parser.parse_args()


# TODO-maybe actually really do some logging
def logError(err):
    logging.exception(err)


def isAGoodResponse(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and
            content_type is not None and content_type.find("html") > -1)


def simpleGet(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if isAGoodResponse(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def getURLS(source):
    result = dict()
    raw_ml = simpleGet(source)
    ml = BeautifulSoup(raw_ml, "lxml")
    ml_str = repr(ml)
    tmp = open("/tmp/riecher", "w")
    tmp.write(ml_str)
    tmp.close()
    tmp = open("/tmp/riecher", "r")
    dump_list = []
    for line in tmp:
        dummy = findall(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|'
            r'(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
        dump_list += dummy
    for elem in dump_list:
        result[elem] = elem
    tmp.close()
    return result


def configNews(config):
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


def main():
    argparser = Argparser()
    config = Config()
    configNews(config)
    urls = getURLS(argparser.args.source)
    for url in urls:
        parser = build(url)
        for article in parser.articles:
            a = Article(article.url)
            try:
                a.download()
                a.parse()
                # print(a.html)
                print(a.text)
            except Exception as e:
                logging.exception(e)


if __name__ == "__main__":
    main()
