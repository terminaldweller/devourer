#!/usr/bin/env python3
# _*_ coding=utf-8 _*_

import argparse
import logging
import subprocess
import sys
import tika
import docker
import os
import nltk
from newspaper import Article, build, Config
from bs4 import BeautifulSoup
from contextlib import closing
from requests import get, Response
from requests.exceptions import RequestException
from re import findall
from readability import Document
from gtts import gTTS
from datetime import datetime as time


class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--source",
                            type=str, help="the url where the \
                            urls to be extracted reside")
        parser.add_argument("--out", type=str,
                            help="the output file", default="")
        parser.add_argument("--singlelink", action="store_true",
                            help="whether the app should work in single-link \
                            meaning only one page's contents will be used \
                            mode", default=False)
        parser.add_argument("--multilink", action="store_true",
                            help="whether the app should work in multi-link \
                            mode meaning the srouce contians a list of links \
                            rather than being the actual source itself",
                            default=False)
        parser.add_argument("--sourcetype", type=str,
                            help="determines the type of the \
                            source.html,text,...")
        parser.add_argument("--pdftomp3", action="store_true",
                            default=False, help="convert pdf to mp3. \
                            source should be the path to a pdf file and\
                            out should be the path to the mp3 output file")
        self.args = parser.parse_args()


# FIXME-maybe actually really do some logging
def logError(err: RequestException) -> None:
    """logs the errors"""
    logging.exception(err)


def isAGoodResponse(resp: Response) -> bool:
    """checks whether the get we sent got a 200 response"""
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and
            content_type is not None)


def simpleGet(url: str) -> bytes:
    """issues a simple get request to download a website"""
    try:
        with closing(get(url, stream=True)) as resp:
            if isAGoodResponse(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def getURLS(source: str) -> dict:
    """extracts the urls from a website"""
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


def configNews(config: Config) -> None:
    """configures newspaper"""
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


def call_from_shell_list(command_list):
    # should probably deprecate this at some point
    if sys.version_info < (3, 7):
        return subprocess.run(command_list, stdout=subprocess.PIPE)
    else:
        return subprocess.run(command_list, capture_output=True)


def pdfToVoice(argparser: Argparser) -> None:
    """main function for converting a pdf to an mp3"""
    TIKA_SERVER_ENDPOINT = "127.0.0.1:9977"
    os.environ["TIKA_SERVER_ENDPOINT"] = TIKA_SERVER_ENDPOINT
    dockerClient = docker.from_env()
    container = dockerClient.containers.run("apache/tika:2.0.0", detach=True,
                                            ports={TIKA_SERVER_ENDPOINT:
                                                   "9998"})
    while True:
        resp = get("http://127.0.0.1:9977")
        if resp.status_code == 200:
            break
        time.sleep(.5)
    rawText = tika.parser.from_file()
    tts = gTTS(rawText['content'])
    tts.save(argparser.args.out)
    container.stop()
    dockerClient.close()


def extractRequirements(textBody: str) -> list:
    result = []
    REQ_KEYWORDS = ["shall", "should", "must", "may", "can", "could"]
    nltk.download("punkt")
    sentences = nltk.sent_tokenize(textBody)
    for sentence in sentences:
        for keyword in REQ_KEYWORDS:
            if sentence.find(keyword) >= 0:
                result.append(sentence)
    return result


def singleLinkMode(argparser: Argparser) -> dict:
    """runs the single-link main function"""
    if argparser.args.sourcetype == "html":
        parser = build(argparser.args.source)
        for article in parser.articles:
            a = Article(article.url)
            try:
                a.download()
                a.parse()
                doc = Document(a.html)
                print(doc.summary())
                extractRequirements(doc.summary())
            except Exception as e:
                logging.exception(e)
    elif argparser.args.sourcetype == "text":
        bytesText = simpleGet(argparser.args.source)
        extractRequirements(bytesText.decode("utf-8"))


def multiLinkMode(argparser: Argparser) -> None:
    """run the multi-link main function"""
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
                doc = Document(a.html)
                print(doc.summary())
                if a.text != '':
                    tts = gTTS(a.text)
                    tts.save(time.today().strftime("%b-%d-%Y-%M-%S-%f")+".mp3")
            except Exception as e:
                logging.exception(e)


def main() -> None:
    argparser = Argparser()
    if argparser.args.singlelink:
        singleLinkMode(argparser)
    elif argparser.args.multilink:
        multiLinkMode(argparser)
    elif argparser.args.pdftomp3:
        pdfToVoice(argparser)
    else:
        pass


if __name__ == "__main__":
    main()
