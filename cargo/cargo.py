#!/usr/bin/env python3
import http.server
import huggingface_hub as hh
import socketserver
import os


# https://huggingface.co/docs/huggingface_hub/how-to-downstream
def download(path: str = ".") -> None:
    bart_pretrained = hh.hf_hub_url(
        "sshleifer/distilbart-cnn-12-6", filename="config.json"
    )
    hh.cached_download(bart_pretrained)


def serve() -> None:
    handler = http.server.SimpleHTTPRequestHandler
    PORT = os.environ["SERVER_PORT"]

    download(os.environ["SERVER_VAULT"])
    with socketserver.TCPServer(("", int(PORT)), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    serve()
