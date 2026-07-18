"""Technique schemas are self-contained and never trigger retrieval."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from distillery.techniques import TechniqueDescriptor, TechniqueError


def test_remote_ref_rejected_without_http_request(external_descriptor) -> None:
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"type":"integer"}')

        def log_message(self, format: str, *args) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = external_descriptor.canonical_payload()
        payload["config_schema"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "seed": {"$ref": (f"http://127.0.0.1:{server.server_port}/remote-schema.json")}
            },
        }
        with pytest.raises(TechniqueError, match="fully inline"):
            TechniqueDescriptor.seal(**payload)
        assert requests == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    "keyword",
    ["$dynamicRef", "$recursiveRef", "$id", "$dynamicAnchor", "default"],
)
def test_retrieval_and_default_keywords_rejected(
    keyword: str,
    external_descriptor,
) -> None:
    payload = external_descriptor.canonical_payload()
    payload["config_schema"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"seed": {keyword: "#/elsewhere"}},
    }
    with pytest.raises(TechniqueError, match="fully inline"):
        TechniqueDescriptor.seal(**payload)
