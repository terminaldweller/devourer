#!/usr/bin/env python3
# _*_ coding=utf-8 _*_

import argparse
import logging
import traceback
from newspaper import Article, build
import fileinput


class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--string", type=str, help="string")
        parser.add_argument("--bool", action="store_true",
                            help="bool", default=False)
        parser.add_argument("--dbg", action="store_true",
                            help="debug", default=False)
        self.args = parser.parse_args()


def main():
    urls = (line for line in fileinput.input())
    for url in urls:
        parser = build(url)
        for article in parser.articles:
            a = Article(article.url)
            try:
                a.download()
                a.parse()
                print(a.text)
            except Exception as e:
                logging.error(traceback.format_exc(e))


if __name__ == "__main__":
    main()
