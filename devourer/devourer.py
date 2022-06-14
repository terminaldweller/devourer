# _*_ coding=utf-8 _*_
"""Personal knowledge aggregator."""

import contextlib
import datetime
import logging
import os
import random
import re
import string
import tempfile
import typing

import bs4  # type:ignore
import fastapi
import gtts  # type:ignore
import newspaper  # type:ignore
import nltk  # type:ignore
import rake_nltk  # type:ignore
import readability  # type:ignore
import refextract  # type:ignore
import requests
import tika  # type:ignore
import transformers
from tika import parser as tparser


# FIXME-maybe actually really do some logging
def log_error(err: str) -> None:
    """Logs the errors."""
    logging.exception(err)


def is_a_good_response(resp: requests.Response) -> bool:
    """Checks whether the get we sent got a 200 response."""
    content_type = resp.headers["Content-Type"].lower()
    return resp.status_code == 200 and content_type is not None


def simple_get(url: str) -> bytes:
    """Issues a simple get request."""
    content = bytes()
    try:
        with contextlib.closing(requests.get(url, stream=True)) as resp:
            if is_a_good_response(resp):
                content = resp.content
    except requests.exceptions.RequestException as e:
        log_error(f"Error during requests to {0} : {1}".format(url, str(e)))
    finally:
        return content


def get_with_params(url: str, params: dict) -> typing.Optional[dict]:
    """Issues a get request with params."""
    try:
        with contextlib.closing(
            requests.get(url, params=params, stream=True)
        ) as resp:
            if is_a_good_response(resp):
                return resp.json()
            return None
    except requests.exceptions.RequestException as e:
        log_error(f"Error during requests to {0} : {1}".format(url, str(e)))
        return None


def get_rand_str(count):
    """Return a random string of the given length."""
    return "".join([random.choice(string.lowercase) for i in range(count)])


def get_urls(source: str, summary: str) -> dict:
    """Extracts the urls from a website."""
    result = {}
    raw_ml = simple_get(source)
    ml = bs4.BeautifulSoup(raw_ml, "lxml")

    rand_tmp = "/tmp/" + get_rand_str(20)
    ml_str = repr(ml)
    tmp = open(rand_tmp, "w", encoding="utf-8")
    tmp.write(ml_str)
    tmp.close()
    tmp = open(rand_tmp, "r", encoding="utf-8")
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


def config_news(config: newspaper.Config) -> None:
    """Configures newspaper."""
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


def sanitize_text(text: str) -> str:
    """Sanitize the strings."""
    text = text.replace("\n", "")
    text = text.replace("\n\r", "")
    text = text.replace('"', "")
    return text


# FIXME-have to decide whether to use files or urls
def pdf_to_voice() -> str:
    """Main function for converting a pdf to an mp3."""
    outfile = str()
    try:
        raw_text = tika.parser.from_file()
        tts = gtts.gTTS(raw_text["content"])
        outfile = get_rand_str(20) + ".mp3"
        tts.save(outfile)
    except Exception as e:
        logging.exception(e)
    finally:
        return outfile


def extract_requirements(text_body: str) -> list:
    """Extract the sentences containing the keywords that denote a requirement.
    the keywords are baed on ISO/IEC directives, part 2:
    https://www.iso.org/sites/directives/current/part2/index.xhtml
    """
    result = []
    req_keywords = [
        "shall",
        "shall not",
        "should",
        "should not",
        "must",
        "may",
        "can",
        "cannot",
    ]
    sentences = nltk.sent_tokenize(text_body)
    for sentence in sentences:
        for keyword in req_keywords:
            if sentence.casefold().find(keyword) >= 0:
                result.append(sanitize_text(sentence))
    return result


def extract_refs(url: str) -> list:
    """Extract the references from an article."""
    refs = []
    try:
        refs = refextract.extract_references_from_url(url)
        return refs
    except Exception as e:
        logging.exception(e)
    finally:
        return refs


def pdf_to_text(url: str) -> str:
    """Convert the PDF file to a string."""
    tika_result = {}
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", delete=True) as tmp_file:
            content = simple_get(url)
            if content is not None:
                tmp_file.write(content)
                tika_result = tparser.from_file(
                    tmp_file.name,
                    serverEndpoint=os.environ["TIKA_SERVER_ENDPOINT"],
                )
    except Exception as e:
        logging.exception(e)
    finally:
        if "content" in tika_result:
            return sanitize_text(tika_result["content"])
        return ""


# TODO-very performance-intensive
def summarize_text(text: str) -> str:
    """Summarize the given text using bart."""
    result = str()
    # TODO-move me later
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

        for i, chunk in enumerate(chunks):
            chunks[i] = "".join(chunk)
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


def summarize_text_v2(text: str) -> str:
    """Text summarization using nltk."""
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = nltk.tokenize.word_tokenize(text)
    freq_table: typing.Dict[str, int] = {}

    for word in words:
        word = word.lower()
        if word in stop_words:
            continue
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    sentences = nltk.tokenize.sent_tokenize(text)
    sentence_value: typing.Dict[str, int] = {}

    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq
                else:
                    sentence_value[sentence] = freq

    sum_values: float = 0
    for sentence, value in sentence_value.items():
        sum_values += value

    average: float = int(sum_values / len(sentence_value))
    summary: str = ""
    for sentence in sentences:
        if (sentence in sentence_value) and (
            sentence_value[sentence] > (1.2 * average)
        ):
            summary += " " + sentence

    return summary


def text_to_audio(text: str) -> str:
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


def get_requirements(url: str, sourcetype: str) -> list:
    """Runs the single-link main function."""
    result = str()
    results = []
    try:
        if sourcetype == "html":
            parser = newspaper.build(url)
            for article in parser.articles:
                art = newspaper.Article(article.url)
                art.download()
                art.parse()
                art.nlp()
                doc = readability.Document(art.html)
                print(doc)
                # print(doc.summary())
                # results = extractRequirements(doc.summary())
                results = extract_requirements(doc)
        elif sourcetype == "text":
            bytes_text = simple_get(url)
            results = extract_requirements(bytes_text.decode("utf-8"))
    except Exception as e:
        logging.exception(e)
    finally:
        print(result)
        # result = "".join(results) + "\n"
        # return result
        return results


# FIXME-summary=bart doesnt work
def summarize_link_to_audio(url: str, summary: str) -> str:
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
            result = sanitize_text(result)
    except Exception as e:
        logging.exception(e)
    finally:
        return result


# FIXME-change my name
def summarize_links_to_audio(origin: str, summary: str) -> str:
    """Summarize a list of urls into audio files."""
    results = []
    result = str()
    try:
        config = newspaper.Config()
        config_news(config)
        urls = get_urls(origin, summary)
        for url in urls:
            results.append(summarize_link_to_audio(url, summary))
    except Exception as e:
        logging.exception(e)
    finally:
        result = "".join(results)
        return result


def search_wikipedia(search_term: str, summary: str) -> str:
    """Search wikipedia for a string and return the url.
    reference: https://www.mediawiki.org/wiki/API:Opensearch.
    """
    result = str()
    try:
        search_params = {
            "action": "opensearch",
            "namespace": "0",
            "search": search_term,
            "limit": "10",
            "format": "json",
        }
        res = get_with_params(os.environ["WIKI_SEARCH_URL"], search_params)
        # FIXME-handle wiki redirects/disambiguations
        if res is not None:
            source = res[3][0]
            result = summarize_link_to_audio(source, summary)
            result = sanitize_text(result)
    except Exception as e:
        logging.exception(e)
    finally:
        return result


def get_audio_from_file(audio_path: str) -> bytes:
    """Returns the contents of a file in binary format."""
    with open(audio_path, "rb") as audio:
        return audio.read()


# TODO- implement me
def get_sentiments(detailed: bool) -> list:
    """Sentiments analysis."""
    results = []
    source = "https://github.com/coinpride/CryptoList"
    urls = simple_get(source)
    classifier = transformers.pipeline("sentiment-analysis")
    for url in urls:
        req_result = simple_get(url)
        results.append(classifier(req_result))
    return results


def get_keywords_from_text(text: str) -> typing.List[str]:
    """Extract keywords out of text."""
    rake_nltk_var = rake_nltk.Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    return rake_nltk_var.get_ranked_phrases()


app = fastapi.FastAPI()

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# https://cheatsheetseries.owasp.org/cheatsheets/REST_Security_Cheat_Sheet.html
@app.middleware("http")
async def add_secure_headers(
    request: fastapi.Request, call_next
) -> fastapi.Response:
    """Adds security headers proposed by OWASP."""
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
    """The pdf manupulation endpoint."""
    if feat == "":
        text = pdf_to_text(url)
        if summarize:
            text = summarize_text_v2(text)
        if audio:
            audio_path = text_to_audio(text)
            return fastapi.Response(
                get_audio_from_file(audio_path) if audio_path != "" else "",
                media_type="audio/mpeg",
            )
        return {
            "Content-Type": "application/json",
            "isOk": bool(text),
            "result": text,
        }
    elif feat == "refs":
        refs = extract_refs(url)
        return {
            "Content-Type": "application/json",
            "isOk": bool(refs),
            "result": refs,
        }
    elif feat == "keyword":
        text = pdf_to_text(url)
        keywords = get_keywords_from_text(text)
        return {
            "Content-Type": "application/json",
            "isOk": bool(keywords),
            "result": keywords,
        }
    else:
        return {
            "Content-Type": "application/json",
            "isOk": False,
            "result": "unknown feature requested",
        }


# TODO- currently not working
@app.get("/mila/tika")
def pdf_to_audio_ep(url: str):
    """Turns a pdf into an audiofile."""
    audio_path = pdf_to_voice()
    return fastapi.Response(
        get_audio_from_file(audio_path) if audio_path != "" else "",
        media_type="audio/mpeg",
    )


@app.get("/mila/reqs")
def extract_reqs_ep(url: str, sourcetype: str = "html"):
    """Extracts the requirements from a given url."""
    result = get_requirements(url, sourcetype)
    return {
        "Content-Type": "application/json",
        "isOK": bool(result),
        "reqs": result,
    }


@app.get("/mila/wiki")
def wiki_search_ep(term: str, summary: str = "none", audio: bool = False):
    """Search and summarizes from wikipedia."""
    text = search_wikipedia(term, summary)
    if audio:
        audio_path = text_to_audio(text)
        return fastapi.Response(
            get_audio_from_file(audio_path) if audio_path != "" else "",
            media_type="audio/mpeg",
        )
    return {
        "Content-Type": "application/json",
        "isOK": bool(text),
        "audio": "",
        "text": text,
    }


@app.get("/mila/summ")
def summarize_ep(url: str, summary: str = "none", audio: bool = False):
    """Summarize and turn the summary into audio."""
    text = summarize_link_to_audio(url, summary)
    if audio:
        audio_path = text_to_audio(text)
        print(audio_path)
        return fastapi.Response(
            get_audio_from_file(audio_path) if audio_path != "" else "",
            media_type="audio/mpeg",
        )
    return {
        "Content-Type": "application/json",
        "isOK": bool(text),
        # "audio": "",
        "text": text,
    }


@app.get("/mila/mila")
def mila_ep(url: str, summary: str = "newspaper", audio: bool = False):
    """Extract all the urls and then summarize and turn into audio."""
    text = summarize_links_to_audio(url, summary)
    if audio:
        audio_path = text_to_audio(text)
        print(audio_path)
        return fastapi.Response(
            get_audio_from_file(audio_path) if audio_path != "" else "",
            media_type="audio/mpeg",
        )
    return {
        "Content-Type": "application/json",
        "isOK": bool(text),
        "audio": "",
        "text": text,
    }


@app.get("/mila/health")
def health_ep():
    """The health endpoint."""
    return {"Content-Type": "application/json", "isOK": True}


@app.get("/mila/robots.txt")
def robots_ep():
    """The robots endpoint."""
    return {
        "Content-Type": "apllication/json",
        "User-Agents": "*",
        "Disallow": "/",
    }
