[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1525a18654274975b8fcfc6992216ad3)](https://www.codacy.com/gh/terminaldweller/devourer/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=terminaldweller/devourer&amp;utm_campaign=Badge_Grade)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/terminaldweller/devourer.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/terminaldweller/devourer/alerts/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# devourer

## About
Devourer is meant to be a sort of a knowledge aggregator though it has quite a long way to get there.<br/>
Right now it can get text out of certain formats, summarize them or not and turn them into audio or not.<br/>
There is a little feature that searches for a term  on wikipedia and one that extracts requirements from RFCs.<br/>

To see what endpoints are present and how to use them or interact with them go to `/docs`. Devourer is powered by fastapi. Everything is in its default location.<br/>

## Live Instance
There is a live instance on https://mila.terminaldweller.com/mila.<br/>
You can look at the docs [here](https://mila.terminalweller.com/docs).<br/>

## How to Run
Run one of the two compose files. The test compose file uses self-signed certificates for local testing since devourer only supports https.<br/>
