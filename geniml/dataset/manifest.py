"""Manifest, cache keying, and provenance.

Two jobs:

- **Cache keying**: a materialized dataset is keyed on (source selection, universe,
  context size) so re-running the same selection is incremental -- it reloads the
  parquet instead of re-tokenizing.
- **Provenance**: emit a small JSON alongside a trained model recording *exactly*
  which bedbase selection, universe, and snapshot date produced it, so an "all of
  bedbase" run is dated and rebuildable.
"""

import hashlib
import json
import os
from typing import List, Optional


def hash_ids(ids: List[str]) -> str:
    """Order-independent hash of a list of item ids (the source selection)."""
    h = hashlib.sha256()
    for item in sorted(str(i) for i in ids):
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def universe_signature(tokenizer) -> str:
    """Stable hash of a tokenizer's vocabulary (identifies the universe)."""
    try:
        vocab = tokenizer.get_vocab()
        blob = json.dumps(vocab, sort_keys=True)
    except Exception:
        blob = repr(getattr(tokenizer, "vocab_size", tokenizer))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def source_manifest(source) -> Optional[List[str]]:
    """Return a source's stable item-id list, or ``None`` if it can't provide one.

    A source opts in to caching by defining ``manifest()``; sources that can't
    (unknown/streaming) return ``None`` and the dataset simply tokenizes fresh.
    """
    fn = getattr(source, "manifest", None)
    if callable(fn):
        return fn()
    return None


def cache_key(source, tokenizer, context_size: int) -> Optional[str]:
    """Compute the cache key for a (source, universe, context_size) combination.

    Returns ``None`` when the source has no manifest (caching disabled).
    """
    ids = source_manifest(source)
    if ids is None:
        return None
    return f"{hash_ids(ids)}.{universe_signature(tokenizer)}.ctx{context_size}"


def write_provenance(
    directory: str,
    *,
    source_ids: Optional[List[str]],
    universe: str,
    context_size: int,
    snapshot_date: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Write a ``dataset_provenance.json`` describing a materialized run.

    Args:
        directory: directory to write into (created if missing) -- typically the
            trained model's output dir.
        source_ids: the exact list of selected item ids (the manifest).
        universe: universe identifier/path used for tokenization.
        context_size: window size used.
        snapshot_date: date the source selection was made (ISO string).
        extra: any additional fields to record (bedbase api, qc filter, etc.).

    Returns:
        str: path to the written JSON file.
    """
    os.makedirs(directory, exist_ok=True)
    payload = {
        "universe": universe,
        "universe_hash": None,
        "context_size": context_size,
        "snapshot_date": snapshot_date,
        "n_items": len(source_ids) if source_ids is not None else None,
        "source_ids_hash": hash_ids(source_ids) if source_ids is not None else None,
        "source_ids": source_ids,
    }
    if extra:
        payload.update(extra)
    path = os.path.join(directory, "dataset_provenance.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path
