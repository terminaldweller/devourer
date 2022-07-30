# _*_ coding=utf-8 _*_
"""Personal knowledge aggregator."""

import contextlib
import datetime
import logging
import os
import tempfile
import typing

import fastapi
import gtts  # type:ignore
import newspaper  # type:ignore
import nltk  # type:ignore
import rake_nltk  # type:ignore
import readability  # type:ignore
import refextract  # type:ignore
import requests  # type:ignore
from tika import parser as tparser  # type:ignore

MODULE_DESCRIPTION = """
Devourer is a lightweight knowledge aggregator.</br>
Right now though, its more of
a personal assistant. It can extract text and summarize it and
 turn it into audio.<br/>
"""
TAGS_METADATA = [
    {
        "name": "/mila/pdf",
        "description": "The PDF endpoint. It accepts urls that contain a "
        "PDF as input. It can summarize the test and turn them into audio.",
    },
    {
        "name": "/mila/reqs",
        "description": "This endpoint accepts a link to a RFC and returns "
        "the requirements it extracts from it.",
    },
    {
        "name": "/mila/wiki",
        "description": "Searches for the given term on wikipedia. Can "
        "optionally summarize the result and turn it into audio.",
    },
    {
        "name": "/mila/summ",
        "description": "The summary endpoint accepts a url as input "
        "that contains an html page. devourer extracts the "
        "__important__ text out of the page and then will either "
        "summarize and optionally turn it into audio.",
    },
    {"name": "/mila/health", "description": "The health endpoint."},
]


# TODO-maybe actually really do some logging
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


def config_news(config: newspaper.Config) -> None:
    """Configures newspaper."""
    config.fetch_images = False
    config.keep_article_html = True
    config.memoize_articles = False
    config.browser_user_agent = "Chrome/91.0.4464.5"


NEWSPAPER_CONFIG = newspaper.Config()
config_news(NEWSPAPER_CONFIG)


def sanitize_text(text: str) -> str:
    """Sanitize the strings."""
    text = text.replace("\n", "")
    text = text.replace("\n\r", "")
    text = text.replace('"', "")
    return text


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
    if "content" in tika_result:
        return sanitize_text(tika_result["content"])
    return ""


# def summarize_text(text: str) -> str:
#     """Summarize the given text using bart."""
#     result = str()
#     transformers_summarizer = transformers.pipeline("summarization")
#     try:
#         sentences = text.split(".")
#         current_chunk = 0
#         max_chunk = 500
#         chunks: list = []

#         for sentence in sentences:
#             if len(chunks) == current_chunk + 1:
#                 if (
#                     len(chunks[current_chunk]) + len(sentence.split(" "))
#                     <= max_chunk
#                 ):
#                     chunks[current_chunk].extend(sentence.split(" "))
#                 else:
#                     current_chunk = +1
#                     chunks.append(sentence.split(" "))
#             else:
#                 chunks.append(sentence.split(" "))
#         print(chunks)

#         for i, chunk in enumerate(chunks):
#             chunks[i] = "".join(chunk)
#         print(chunks)

#         summaries = transformers_summarizer(
#             chunks, max_length=50, min_length=30, do_sample=False
#         )

#         result = "".join([summary["summary_text"] for summary in summaries])
#         print(result)
#     except Exception as e:
#         logging.exception(e)
#     return result


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
    return path


def get_requirements(url: str, sourcetype: str) -> list:
    """Runs the single-link main function."""
    results = []
    try:
        if sourcetype == "html":
            parser = newspaper.build(url, NEWSPAPER_CONFIG)
            for article in parser.articles:
                art = newspaper.Article(
                    config=NEWSPAPER_CONFIG, url=article.url
                )
                art.download()
                art.parse()
                art.nlp()
                doc = readability.Document(art.html)
                results = extract_requirements(doc)
        elif sourcetype == "text":
            bytes_text = simple_get(url)
            results = extract_requirements(bytes_text.decode("utf-8"))
    except Exception as e:
        logging.exception(e)
    return results


def summarize_link_to_audio(url: str, summary: str) -> str:
    """Summarizes the text inside a given url into audio."""
    result = str()
    try:
        article = newspaper.Article(config=NEWSPAPER_CONFIG, url=url)
        article.download()
        article.parse()
        if summary == "newspaper":
            article.nlp()
            result = article.summary
        elif summary == "none":
            result = article.text
        elif summary == "nltk":
            result = summarize_text_v2(article.text)
        else:
            print("invalid option for summary type.")
        if result != "":
            result = sanitize_text(result)
    except Exception as e:
        logging.exception(e)
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
    return result


def get_audio_from_file(audio_path: str) -> bytes:
    """Returns the contents of a file in binary format."""
    with open(audio_path, "rb") as audio:
        return audio.read()


def get_keywords_from_text(text: str) -> typing.List[str]:
    """Extract keywords out of text."""
    rake_nltk_var = rake_nltk.Rake()
    rake_nltk_var.extract_keywords_from_text(text)
    return rake_nltk_var.get_ranked_phrases()


app = fastapi.FastAPI(
    title="Devourer",
    description=MODULE_DESCRIPTION,
    contact={
        "name": "farzad sadeghi",
        "url": "https://github.com/terminaldweller/devourer",
        "email": "thabogre@gmail.com",
    },
    license_info={
        "name": "GPL v3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.en.html",
    },
    openapi_tags=TAGS_METADATA,
)

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


@app.get("/mila/pdf", tags=["/mila/pdf"])
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
    if feat == "refs":
        refs = extract_refs(url)
        return {
            "Content-Type": "application/json",
            "isOk": bool(refs),
            "result": refs,
        }
    if feat == "keyword":
        text = pdf_to_text(url)
        keywords = get_keywords_from_text(text)
        return {
            "Content-Type": "application/json",
            "isOk": bool(keywords),
            "result": keywords,
        }
    return {
        "Content-Type": "application/json",
        "isOk": False,
        "result": "unknown feature requested",
    }


@app.get("/mila/reqs", tags=["/mila/reqs"])
def extract_reqs_ep(url: str, sourcetype: str = "html"):
    """Extracts the requirements from a given url."""
    result = get_requirements(url, sourcetype)
    return {
        "Content-Type": "application/json",
        "isOK": bool(result),
        "reqs": result,
    }


@app.get("/mila/wiki", tags=["/mila/wiki"])
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


@app.get("/mila/summ", tags=["/mila/summ"])
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
        "text": text,
    }


@app.get("/mila/health", tags=["/mila/health"])
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
