# _*_ coding=utf-8 _*_

import contextlib
import datetime
import logging
import os
import random
import re
import string
import tempfile
import typing

import bs4
import fastapi
import gtts
import newspaper
import nltk
import readability
import refextract
import requests
import tika
import transformers
from tika import parser as tparser


# FIXME-maybe actually really do some logging
def logError(err: str) -> None:
    """Logs the errors."""
    logging.exception(err)


def isAGoodResponse(resp: requests.Response) -> bool:
    """Checks whether the get we sent got a 200 response."""
    content_type = resp.headers["Content-Type"].lower()
    return resp.status_code == 200 and content_type is not None


def simpleGet(url: str) -> bytes:
    """Issues a simple get request."""
    content = bytes()
    try:
        with contextlib.closing(requests.get(url, stream=True)) as resp:
            if isAGoodResponse(resp):
                content = resp.content
    except requests.exceptions.RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
    finally:
        return content


def getWithParams(url: str, params: dict) -> typing.Optional[dict]:
    """Issues a get request with params."""
    try:
        with contextlib.closing(
            requests.get(url, params=params, stream=True)
        ) as resp:
            if isAGoodResponse(resp):
                return resp.json()
            else:
                return None
    except requests.exceptions.RequestException as e:
        logError("Error during requests to {0} : {1}".format(url, str(e)))
        return None


def getRandStr(n):
    """Return a random string of the given length."""
    return "".join([random.choice(string.lowercase) for i in range(n)])


def getURLS(source: str, summary: str) -> dict:
    """Extracts the urls from a website."""
    result = dict()
    raw_ml = simpleGet(source)
    ml = bs4.BeautifulSoup(raw_ml, "lxml")

    rand_tmp = "/tmp/" + getRandStr(20)
    ml_str = repr(ml)
    tmp = open(rand_tmp, "w")
    tmp.write(ml_str)
    tmp.close()
    tmp = open(rand_tmp, "r")
    url_list = []
    for line in tmp:
        url = re.findall(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|"
            r"(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            line,
        )
        url_list += url
    for elem in url_list:
        result[elem] = elem
    tmp.close()
    return result


def configNews(config: newspaper.Config) -> None:
    """Configures newspaper."""
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


def sanitizeText(text: str) -> str:
    """Sanitize the strings."""
    text = text.replace("\n", "")
    text = text.replace("\n\r", "")
    text = text.replace('"', "")
    return text


# FIXME-have to decide whether to use files or urls
def pdfToVoice() -> str:
    """Main function for converting a pdf to an mp3."""
    outfile = str()
    try:
        rawText = tika.parser.from_file()
        tts = gtts.gTTS(rawText["content"])
        outfile = getRandStr(20) + ".mp3"
        tts.save(outfile)
    except Exception as e:
        logging.exception(e)
    finally:
        return outfile


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
    sentences = nltk.sent_tokenize(textBody)
    for sentence in sentences:
        for keyword in REQ_KEYWORDS:
            if sentence.casefold().find(keyword) >= 0:
                result.append(sanitizeText(sentence))
    return result


def extractRefs(url: str) -> list:
    refs = list()
    try:
        refs = refextract.extract_references_from_url(url)
        return refs
    except Exception as e:
        logging.exception(e)
    finally:
        return refs


def pdfToText(url: str) -> str:
    """Convert the PDF file to a string."""
    tikaResult = dict()
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=True) as tmpFile:
            content = simpleGet(url)
            if content is not None:
                tmpFile.write(content)
                tikaResult = tparser.from_file(
                    tmpFile.name,
                    serverEndpoint=os.environ["TIKA_SERVER_ENDPOINT"],
                )
                # print(tikaResult["metadata"])
                # print(tikaResult["content"])
    except Exception as e:
        logging.exception(e)
    finally:
        if "content" in tikaResult:
            return sanitizeText(tikaResult["content"])
        else:
            return ""


# FIXME doesnt work for long texts
def summarizeText(text: str) -> str:
    """Summarize the given text using bart."""
    result = str()
    # TODO move me later
    transformers_summarizer = transformers.pipeline("summarization")
    try:
        sentences = text.split(".")
        current_chunk = 0
        max_chunk = 500
        chunks: list = []

        for sentence in sentences:
            if len(chunks) == current_chunk + 1:
                if (
                    len(chunks[current_chunk]) + len(sentence.split(" "))
                    <= max_chunk
                ):
                    chunks[current_chunk].extend(sentence.split(" "))
                else:
                    current_chunk = +1
                    chunks.append(sentence.split(" "))
            else:
                chunks.append(sentence.split(" "))
        print(chunks)

        for chunk_id in range(len(chunks)):
            chunks[chunk_id] = "".join(chunks[chunk_id])
        print(chunks)

        summaries = transformers_summarizer(
            chunks, max_length=50, min_length=30, do_sample=False
        )

        result = "".join([summary["summary_text"] for summary in summaries])
        print(result)
    except Exception as e:
        logging.exception(e)
    finally:
        return result


def summarizeText_v2(text: str) -> str:
    pass


def textToAudio(text: str) -> str:
    """Transform the given text into audio."""
    path = str()
    try:
        time_str = datetime.datetime.today().strftime("%b-%d-%Y-%M-%S-%f")
        tts = gtts.gTTS(text)
        tts.save(os.environ["AUDIO_DUMP_DIR"] + "/" + time_str + ".mp3")
        path = os.environ["AUDIO_DUMP_DIR"] + "/" + time_str + ".mp3"
    except Exception as e:
        logging.exception(e)
    finally:
        return path


def getRequirements(url: str, sourcetype: str) -> list:
    """Runs the single-link main function."""
    result = str()
    results = list()
    try:
        if sourcetype == "html":
            parser = newspaper.build(url)
            for article in parser.articles:
                a = newspaper.Article(article.url)
                a.download()
                a.parse()
                a.nlp()
                doc = readability.Document(a.html)
                print(doc)
                # print(doc.summary())
                # results = extractRequirements(doc.summary())
                results = extractRequirements(doc)
        elif sourcetype == "text":
            bytesText = simpleGet(url)
            results = extractRequirements(bytesText.decode("utf-8"))
    except Exception as e:
        logging.exception(e)
    finally:
        print(result)
        # result = "".join(results) + "\n"
        # return result
        return results


# FIXME-summary=bart doesnt work
def summarizeLinkToAudio(url: str, summary: str) -> str:
    """Summarizes the text inside a given url into audio."""
    result = str()
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        if summary == "newspaper":
            article.nlp()
            result = article.summary
        elif summary == "none":
            result = article.text
        elif summary == "bart":
            result = article.text
        else:
            print("invalid option for summary type.")
        if result != "":
            result = sanitizeText(result)
    except Exception as e:
        logging.exception(e)
    finally:
        return result


# FIXME-change my name
def summarizeLinksToAudio(url: str, summary: str) -> str:
    """Summarize a list of urls into audio files."""
    results = list()
    result = str()
    try:
        config = newspaper.Config()
        configNews(config)
        urls = getURLS(url, summary)
        for url in urls:
            results.append(summarizeLinkToAudio(url, summary))
    except Exception as e:
        logging.exception(e)
    finally:
        result = "".join(results)
        return result


def searchWikipedia(search_term: str, summary: str) -> str:
    """Search wikipedia for a string and return the url.
    reference: https://www.mediawiki.org/wiki/API:Opensearch
    """
    result = str()
    try:
        searchParmas = {
            "action": "opensearch",
            "namespace": "0",
            "search": search_term,
            "limit": "10",
            "format": "json",
        }
        res = getWithParams(os.environ["WIKI_SEARCH_URL"], searchParmas)
        # FIXME-handle wiki redirects/disambiguations
        if res is not None:
            source = res[3][0]
            result = summarizeLinkToAudio(source, summary)
            result = sanitizeText(result)
    except Exception as e:
        logging.exception(e)
    finally:
        return result


def getAudioFromFile(audio_path: str) -> bytes:
    """Returns the contents of a file in binary format."""
    with open(audio_path, "rb") as audio:
        return audio.read()


"""
def getSentiments(detailed: bool) -> list:
    results = list()
    SOURCE = "https://github.com/coinpride/CryptoList"
    urls = simpleGet(SOURCE)
    classifier = transformers.pipeline("sentiment-analysis")
    for url in urls:
        req_result = simpleGet(url)
        results.append(classifier(req_result))
    return results
"""


app = fastapi.FastAPI()

nltk.download("punkt")


# https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html
@app.middleware("http")
async def addSecureHeaders(
    request: fastapi.Request, call_next
) -> fastapi.Response:
    """adds security headers proposed by OWASP."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src-https"
    response.headers["Strict-Transport-Security"] = "max-age=63072000"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Access-Control-Allow-Methods"] = "GET,OPTIONS"
    return response


@app.get("/mila/pdf")
def pdf_ep(
    url: str, feat: str = "", audio: bool = False, summarize: bool = False
):
    """the pdf manupulation endpoint."""
    if feat == "":
        text = pdfToText(url)
        if summarize:
            text = summarizeText(text)
        if audio:
            audio_path = textToAudio(text)
            return fastapi.Response(
                getAudioFromFile(audio_path) if audio_path != "" else "",
                media_type="audio/mpeg",
            )
        return {
            "Content-Type": "application/json",
            "isOk": True if text != "" else False,
            "result": text,
        }
    elif feat == "refs":
        refs = extractRefs(url)
        return {
            "Content-Type": "application/json",
            "isOk": True if refs is not None else False,
            "result": refs,
        }


@app.get("/mila/tika")
def pdf_to_audio_ep(url: str):
    """turns a pdf into an audiofile."""
    audio_path = pdfToVoice()
    return fastapi.Response(
        getAudioFromFile(audio_path) if audio_path != "" else "",
        media_type="audio/mpeg",
    )


@app.get("/mila/reqs")
def extract_reqs_ep(url: str, sourcetype: str = "html"):
    """extracts the requirements from a given url."""
    result = getRequirements(url, sourcetype)
    return {
        "Content-Type": "application/json",
        "isOK": True if result is not None else False,
        "reqs": result,
    }


@app.get("/mila/wiki")
def wiki_search_ep(term: str, summary: str = "none", audio: bool = False):
    """search and summarizes from wikipedia."""
    text = searchWikipedia(term, summary)
    if audio:
        audio_path = textToAudio(text)
        return fastapi.Response(
            getAudioFromFile(audio_path) if audio_path != "" else "",
            media_type="audio/mpeg",
        )
    else:
        return {
            "Content-Type": "application/json",
            "isOK": True if text != "" else False,
            "audio": "",
            "text": text,
        }


@app.get("/mila/summ")
def summarize_ep(url: str, summary: str = "none", audio: bool = False):
    """summarize and turn the summary into audio."""
    text = summarizeLinkToAudio(url, summary)
    if audio:
        audio_path = textToAudio(text)
        print(audio_path)
        return fastapi.Response(
            getAudioFromFile(audio_path) if audio_path != "" else "",
            media_type="audio/mpeg",
        )
    else:
        return {
            "Content-Type": "application/json",
            "isOK": True if text != "" else False,
            # "audio": "",
            "text": text,
        }


@app.get("/mila/mila")
def mila_ep(url: str, summary: str = "newspaper", audio: bool = False):
    """extract all the urls and then summarize and turn into audio."""
    text = summarizeLinksToAudio(url, summary)
    if audio:
        audio_path = textToAudio(text)
        print(audio_path)
        return fastapi.Response(
            getAudioFromFile(audio_path) if audio_path != "" else "",
            media_type="audio/mpeg",
        )
    else:
        return {
            "Content-Type": "application/json",
            "isOK": True if text != "" else False,
            "audio": "",
            "text": text,
        }


@app.get("/mila/health")
def health_ep():
    return {"Content-Type": "application/json", "isOK": True}


@app.get("/mila/robots.txt")
def robots_ep():
    return {
        "Content-Type": "apllication/json",
        "User-Agents": "*",
        "Disallow": "/",
    }
