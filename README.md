[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1525a18654274975b8fcfc6992216ad3)](https://www.codacy.com/gh/terminaldweller/devourer/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=terminaldweller/devourer&amp;utm_campaign=Badge_Grade)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/terminaldweller/devourer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/terminaldweller/devourer/alerts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# devourer
devourer is an api server that currently has the following endpoints and does the following things:<br/>

## /summ
```sh
https://localhost:19019/mila/summ?url=https://dilipkumar.medium.com/standalone-mongodb-on-kubernetes-cluster-19e7b5896b27&summary=newspaper&audio=true
```
The `/summ` endpoint optionally summarizes the article and can also optionally send the article as an audio file.<br/>
The parameters are `url`,`summary` tells the server which summarization method to use. the last parameter `audio` tells the server whether to just send the text result or an audio equivalent.<br/>

## /wiki
```sh
https://localhost:19019/mila/wiki?term=iommu&summary=newspaper&audio=true
```
Searches wikipedia for the given `term` parameter. Like other endpoints, can optionally summarize the result and turn it into audio with `summary` and `audio` parameters.<br/>

## /reqs
```sh
https://localhost:19019/mila/reqs?url=https://www.ietf.org/rfc/rfc2865.txt&sourcetype=text
```
Extracts the requirements from the contents inside a given url. The `sourcetype` parameter tells the server how to interpret the url contents. currently only `text` and `html` are supported as valid values.<br/>

## Usage Example

Using FastAPI now so just check the docs.
