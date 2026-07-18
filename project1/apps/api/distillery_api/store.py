"""Immutable filesystem-backed metadata store. Resources are write-once JSON
documents; corrections mint new IDs."""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from distillery.contracts.errors import NotFound

DATA_DIR = Path(os.environ.get("DISTILLERY_DATA_DIR", "/data"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class Store:
    def __init__(self, root: Path | None = None):
        self.root = root or DATA_DIR
        for sub in ("datasets", "runs", "artifacts", "reports"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, rid: str) -> Path:
        return self.root / kind / f"{rid}.json"

    def put(self, kind: str, rid: str, doc: dict, *, overwrite: bool = False) -> None:
        p = self._path(kind, rid)
        if p.exists() and not overwrite:
            raise FileExistsError(rid)
        p.write_text(json.dumps(doc, indent=2, default=str))

    def get(self, kind: str, rid: str) -> dict:
        p = self._path(kind, rid)
        if not p.exists():
            raise NotFound(f"{kind[:-1]} {rid} not found")
        return json.loads(p.read_text())

    def list(self, kind: str) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.root / kind).glob("*.json"))]

    def append_event(self, run_id: str, event: dict) -> None:
        p = self.root / "runs" / f"{run_id}.events.jsonl"
        with p.open("a") as f:
            f.write(json.dumps({"at": now_iso(), **event}) + "\n")

    def blob_dir(self, kind: str, rid: str) -> Path:
        d = self.root / kind / rid
        d.mkdir(parents=True, exist_ok=True)
        return d
