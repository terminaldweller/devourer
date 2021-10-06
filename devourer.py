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


WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


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
        parser.add_argument("--summary", type=str, default="newspaper",
                            help="which summary type to use. currently we \
                            have newspaper, bart and none.")
        parser.add_argument("--search", type=str,
                            default="", help="the search query")
        self.args = parser.parse_args()


# FIXME-maybe actually really do some logging
def logError(err: RequestException) -> None:
    """logs the errors."""
    logging.exception(err)


def isAGoodResponse(resp: Response) -> bool:
    """checks whether the get we sent got a 200 response."""
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 and
            content_type is not None)


def simpleGet(url: str) -> bytes:
    """issues a simple get request."""
    try:
        with closing(get(url, stream=True)) as resp:
            if isAGoodResponse(resp):
                return resp.content
            else:
                return None
    except RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def getWithParams(url: str, params: dict) -> dict:
    """issues a get requesti with params."""
    try:
        with closing(get(url, params=params, stream=True)) as resp:
            if isAGoodResponse(resp):
                return resp.json()
            else:
                return None
    except RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def getURLS(source: str) -> dict:
    """extracts the urls from a website."""
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
    """configures newspaper."""
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


# TODO-should probably deprecate this at some point
def call_from_shell_list(command_list: list):
    """run a shell command given a list of command/arguments."""
    if sys.version_info < (3, 7):
        return subprocess.run(command_list, stdout=subprocess.PIPE)
    else:
        return subprocess.run(command_list, capture_output=True)


def pdfToVoice(argparser: Argparser) -> None:
    """main function for converting a pdf to an mp3."""
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
    """extract the sentences containing the keywords
     that denote a requirement."""
    result = []
    REQ_KEYWORDS = ["shall", "should", "must", "may", "can", "could"]
    nltk.download("punkt")
    sentences = nltk.sent_tokenize(textBody)
    for sentence in sentences:
        for keyword in REQ_KEYWORDS:
            if sentence.find(keyword) >= 0:
                result.append(sentence)
    return result


def summarizeText(text: str) -> str:
    """summarize the given text using bart."""
    from transformers import BartTokenizer, BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer([text],
                       max_length=1024, return_tensors='pt')
    summary_ids = model.generate(
        inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    return([tokenizer.decode(g,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)
            for g in summary_ids])


def textToAudio(text: str) -> None:
    """transform the given text into audio."""
    tts = gTTS(text)
    tts.save(time.today().strftime("%b-%d-%Y-%M-%S-%f")+".mp3")


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


def summarizeLinkToAudio(argparser: Argparser) -> None:
    """summarizes the text inside a given url into audio."""
    try:
        article = Article(argparser.args.source)
        article.download()
        article.parse()
        if argparser.args.summary == "newspaper":
            article.nlp()
            textToAudio(article.summary)
        elif argparser.args.summary == "none":
            textToAudio(article.text)
        elif argparser.args.summary == "bart":
            textToAudio(summarizeText(article.text))
        else:
            print("invalid option for summry type.")
    except Exception as e:
        logging.exception(e)


def summarizeLinksToAudio(argparser: Argparser) -> None:
    """summarize a list of urls into audio files."""
    config = Config()
    configNews(config)
    urls = getURLS(argparser.args.source)
    for url in urls:
        summarizeLinkToAudio(url)


def searchWikipedia(argparser: Argparser) -> str:
    """search wikipedia for a string and return the url."""
    searchParmas = {
        "action": "opensearch",
        "namespace": "0",
        "search": argparser.args.search,
        "limit": "10",
        "format": "json"
    }
    res = getWithParams(WIKIPEDIA_SEARCH_URL, searchParmas)
    print(res)
    argparser.args.source = res[3][0]
    summarizeLinkToAudio(argparser)


def main() -> None:
    argparser = Argparser()
    if argparser.args.singlelink:
        summarizeLinkToAudio(argparser)
    elif argparser.args.multilink:
        summarizeLinksToAudio(argparser)
    elif argparser.args.pdftomp3:
        pdfToVoice(argparser)
    elif argparser.args.search:
        searchWikipedia(argparser)
    else:
        pass


if __name__ == "__main__":
    main()
