"""Durable storage on top of JSON.

Lumen is intentionally file-backed rather than server-backed: a collection is a
single self-describing JSON document, so it can be committed to git, shipped in
a model artefact, or copied across machines with no database process. The on-disk
shape is versioned so the loader can reject incompatible files loudly.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

FORMAT = "lumen-collection"
VERSION = 1
SUFFIX = ".lumen.json"


def _encode(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _decode(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if data.get("format") != FORMAT:
        raise ValueError("not a Lumen collection file")
    if data.get("version", 0) > VERSION:
        raise ValueError(
            f"collection version {data.get('version')} is newer than supported {VERSION}"
        )
    return data


def write_collection_file(path: str, data: Dict[str, Any]) -> None:
    """Atomically write a collection document."""
    payload = _encode(data)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
    os.replace(tmp, path)


def read_collection_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return _decode(fh.read())


def collection_path(directory: str, name: str) -> str:
    return os.path.join(directory, name + SUFFIX)


def list_collection_files(directory: str) -> List[str]:
    """Return absolute paths of every *.lumen.json file in ``directory``."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(SUFFIX)
    )


def make_collection_dict(
    name: str,
    dim: int,
    metric: str,
    index_type: str,
    index_params: Dict[str, Any],
    records: List[List[Any]],
) -> Dict[str, Any]:
    """Assemble the serialisable representation of a collection.

    ``records`` is a list of ``[id, vector, metadata]`` triples.
    """
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "dim": dim,
        "metric": metric,
        "index_type": index_type,
        "index_params": index_params,
        "vectors": records,
    }
