#!/usr/bin/env python3
# _*_ coding=utf-8 _*_

import argparse
import logging
import tika
import docker
import os
import nltk
import random
import string
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
        parser.add_argument(
            "--source",
            type=str,
            help="the url where the \
                            urls to be extracted reside",
            default="",
        )
        parser.add_argument(
            "--out",
            type=str,
            help="the output file name if it applies",
            default="",
        )
        parser.add_argument(
            "--singlelink",
            action="store_true",
            help="whether the app should work in single-link \
                            meaning only one page's contents will be used \
                            mode",
            default=False,
        )
        parser.add_argument(
            "--multilink",
            action="store_true",
            help="whether the app should work in multi-link \
                            mode meaning the srouce contians a list of links \
                            rather than being the actual source itself",
            default=False,
        )
        parser.add_argument(
            "--sourcetype",
            type=str,
            help="determines the type of the \
                            source:html,text,...",
            default="html",
        )
        parser.add_argument(
            "--pdftomp3",
            action="store_true",
            default=False,
            help="convert pdf to mp3. \
                            source should be the path to a pdf file and\
                            out should be the path to the mp3 output file",
        )
        parser.add_argument(
            "--summary",
            type=str,
            default="newspaper",
            help="which summary type to use. currently we \
                            have newspaper, bart and none.",
        )
        parser.add_argument(
            "--search", type=str, default="", help="the string to search for"
        )
        self.args = parser.parse_args()


# FIXME-maybe actually really do some logging
def logError(err: RequestException) -> None:
    """Logs the errors."""
    logging.exception(err)


def isAGoodResponse(resp: Response) -> bool:
    """Checks whether the get we sent got a 200 response."""
    content_type = resp.headers["Content-Type"].lower()
    return resp.status_code == 200 and content_type is not None


def simpleGet(url: str) -> bytes:
    """Issues a simple get request."""
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
    """Issues a get requesti with params."""
    try:
        with closing(get(url, params=params, stream=True)) as resp:
            if isAGoodResponse(resp):
                return resp.json()
            else:
                return None
    except RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def getRandStr(n):
    """Return a random string of the given length."""
    return "".join([random.choice(string.lowercase) for i in range(n)])


def getURLS(source: str) -> dict:
    """Extracts the urls from a website."""
    result = dict()
    raw_ml = simpleGet(source)
    ml = BeautifulSoup(raw_ml, "lxml")

    rand_tmp = "/tmp/" + getRandStr(20)
    ml_str = repr(ml)
    tmp = open(rand_tmp, "w")
    tmp.write(ml_str)
    tmp.close()
    tmp = open(rand_tmp, "r")
    url_list = []
    for line in tmp:
        url = findall(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            line,
        )
        url_list += url
    for elem in url_list:
        result[elem] = elem
    tmp.close()
    return result


def configNews(config: Config) -> None:
    """Configures newspaper."""
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


def pdfToVoice(argparser: Argparser) -> None:
    """Main function for converting a pdf to an mp3."""
    TIKA_SERVER_ENDPOINT = "127.0.0.1:9977"
    os.environ["TIKA_SERVER_ENDPOINT"] = TIKA_SERVER_ENDPOINT
    dockerClient = docker.from_env()
    container = dockerClient.containers.run(
        "apache/tika:2.0.0", detach=True, ports={TIKA_SERVER_ENDPOINT: "9998"}
    )
    while True:
        resp = get("http://127.0.0.1:9977")
        if resp.status_code == 200:
            break
        time.sleep(0.5)
    rawText = tika.parser.from_file()
    tts = gTTS(rawText["content"])
    tts.save(argparser.args.out)
    container.stop()
    dockerClient.close()


def extractRequirements(textBody: str) -> list:
    """Extract the sentences containing the keywords that denote a requirement.

    the keywords are baed on ISO/IEC directives, part 2:
    https://www.iso.org/sites/directives/current/part2/index.xhtml
    """
    result = []
    REQ_KEYWORDS = [
        "shall",
        "shall not",
        "should",
        "should not",
        "must",
        "may",
        "can",
        "cannot",
    ]
    nltk.download("punkt")
    sentences = nltk.sent_tokenize(textBody)
    for sentence in sentences:
        for keyword in REQ_KEYWORDS:
            if sentence.find(keyword) >= 0:
                result.append(sentence)
    return result


def summarizeText(text: str) -> str:
    """Summarize the given text using bart."""
    import transformers

    model = transformers.BartForConditionalGeneration.from_pretrained(
        "facebook/bart-large-cnn"
    )
    tokenizer = transformers.BartTokenizer.from_pretrained(
        "facebook/bart-large-cnn"
    )
    inputs = tokenizer([text], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"], num_beams=4, max_length=5, early_stopping=True
    )
    return [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for g in summary_ids
    ]


def textToAudio(text: str) -> None:
    """Transform the given text into audio."""
    tts = gTTS(text)
    tts.save(time.today().strftime("%b-%d-%Y-%M-%S-%f") + ".mp3")


def singleLinkMode(argparser: Argparser) -> dict:
    """Runs the single-link main function."""
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
    """Summarizes the text inside a given url into audio."""
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
    """Summarize a list of urls into audio files."""
    config = Config()
    configNews(config)
    urls = getURLS(argparser.args.source)
    for url in urls:
        summarizeLinkToAudio(url)


def searchWikipedia(argparser: Argparser) -> str:
    """Search wikipedia for a string and return the url.

    reference: https://www.mediawiki.org/wiki/API:Opensearch
    """
    searchParmas = {
        "action": "opensearch",
        "namespace": "0",
        "search": argparser.args.search,
        "limit": "10",
        "format": "json",
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
