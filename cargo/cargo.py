#!/usr/bin/env python3
"""Cargo is meant to server as a file/server downloader service."""
import http.server
import os
import socketserver

import huggingface_hub as hh


# https://huggingface.co/docs/huggingface_hub/how-to-downstream
def download(path: str = ".") -> None:
    """Download the required models from huggingface."""
    bart_pretrained = hh.hf_hub_url(
        "sshleifer/distilbart-cnn-12-6", filename="config.json"
    )
    hh.cached_download(bart_pretrained)


def serve() -> None:
    """Startup a simple http file server."""
    handler = http.server.SimpleHTTPRequestHandler
    port_number = os.environ["SERVER_PORT"]

    download(os.environ["SERVER_VAULT"])
    with socketserver.TCPServer(("", int(port_number)), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    serve()
