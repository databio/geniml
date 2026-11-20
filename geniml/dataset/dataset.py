"""Source-agnostic tokenized training dataset.

:class:`TokenizedRegionDataset` is the join between the *read* half (a
:class:`~geniml.dataset.RegionSetSource`) and the *train* half (a tokenizer + a
HuggingFace ``Trainer``). It tokenizes and windows every region set from a source
into training rows (``input_ids`` + preserved metadata), in one of two modes:

- ``materialize`` -- tokenize once into an on-disk parquet/Arrow dataset and return
  a ``datasets.Dataset``. Cache is keyed on (source selection, universe, context
  size), so re-running the same selection reloads instead of re-tokenizing. Best for
  multi-epoch training.
- ``stream`` -- a ``torch.utils.data.IterableDataset`` that tokenizes on the fly,
  with an optional ``.gtok`` cache so epoch 2+ skips re-tokenization. Best for "all
  of bedbase, always current, no giant parquet".

Rows match what the atacformer RTD collator expects (``input_ids``; masks come from
the collator). ``datasets`` / ``torch`` are imported lazily so importing the source
and tokenization halves works on a base install.
"""

import os
import random
from logging import getLogger
from typing import Dict, List, Optional

from .manifest import cache_key
from .source import RegionSetSource
from .tokenize import sample_and_remove, tokenize_regionset

_LOGGER = getLogger("geniml.dataset")


class TokenizedRegionDataset:
    """Tokenize + window any :class:`RegionSetSource` into a training dataset.

    Args:
        source: where region sets come from (a :class:`RegionSetSource`).
        tokenizer: a ``gtars`` tokenizer / ``TrainingTokenizer`` bound to a universe.
        context_size: tokens per training window.
        mode: ``"materialize"`` (default) or ``"stream"``.
        cache_dir: directory for the materialized parquet cache and/or gtok cache.
        max_windows_per_file: skip a file producing more than this many windows.
        num_proc: worker processes for materialize-mode tokenization. Parallelism
            requires ``universe`` (workers rebuild the tokenizer) and path-backed
            items; otherwise it falls back to sequential.
        seed: RNG seed for reproducible windowing.
        drop_unk: drop ``unk`` tokens before windowing.
        universe: universe path/id used to rebuild the tokenizer in worker processes
            (enables ``num_proc > 1``).
    """

    def __init__(
        self,
        source: RegionSetSource,
        tokenizer,
        context_size: int = 8192,
        mode: str = "materialize",
        cache_dir: Optional[str] = None,
        max_windows_per_file: int = 10,
        num_proc: int = 1,
        seed: int = 42,
        drop_unk: bool = True,
        universe: Optional[str] = None,
    ):
        if mode not in ("materialize", "stream"):
            raise ValueError(f"mode must be 'materialize' or 'stream', got {mode!r}")
        self.source = source
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.mode = mode
        self.cache_dir = cache_dir
        self.max_windows_per_file = max_windows_per_file
        self.num_proc = max(1, num_proc)
        self.seed = seed
        self.drop_unk = drop_unk
        self.universe = universe

    # -- public API -------------------------------------------------------

    def build(self):
        """Build the dataset according to ``mode``.

        Returns:
            A ``datasets.Dataset`` (materialize) or a
            :class:`StreamingTokenizedRegionDataset` (stream).
        """
        if self.mode == "materialize":
            return self.materialize()
        return self.stream()

    def materialize(self):
        """Tokenize the whole source into an on-disk ``datasets.Dataset`` (cached)."""
        try:
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "materialize mode requires the `datasets` package (pip install geniml[ml])."
            ) from exc

        out_path = self._cache_parquet_path()
        if out_path and os.path.exists(out_path):
            _LOGGER.info("Loading cached tokenized dataset from %s", out_path)
            return Dataset.from_parquet(out_path)

        rows = self._tokenize_all()
        _LOGGER.info("Tokenized %d windows from source", len(rows))
        ds = Dataset.from_list(rows)

        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ds.to_parquet(out_path)
            _LOGGER.info("Wrote tokenized cache to %s", out_path)
        return ds

    def stream(self) -> "StreamingTokenizedRegionDataset":
        """Return a streaming ``IterableDataset`` over the source."""
        return StreamingTokenizedRegionDataset(
            source=self.source,
            tokenizer=self.tokenizer,
            context_size=self.context_size,
            max_windows_per_file=self.max_windows_per_file,
            cache_dir=self.cache_dir,
            seed=self.seed,
            drop_unk=self.drop_unk,
        )

    # -- internals --------------------------------------------------------

    def _cache_parquet_path(self) -> Optional[str]:
        if not self.cache_dir:
            return None
        key = cache_key(self.source, self.tokenizer, self.context_size)
        if key is None:
            return None
        return os.path.join(self.cache_dir, f"tokenized_{key}.parquet")

    def _tokenize_all(self) -> List[Dict]:
        if self.num_proc > 1:
            parallel_rows = self._tokenize_parallel()
            if parallel_rows is not None:
                return parallel_rows
            _LOGGER.warning(
                "num_proc>1 requested but source is not path-backed or `universe` "
                "was not given; falling back to sequential tokenization."
            )
        return self._tokenize_sequential()

    def _tokenize_sequential(self) -> List[Dict]:
        rows: List[Dict] = []
        rng = random.Random(self.seed)
        for region_set, meta in self.source:
            windows = tokenize_regionset(
                region_set,
                self.tokenizer,
                context_size=self.context_size,
                max_windows=self.max_windows_per_file,
                rng=rng,
                drop_unk=self.drop_unk,
            )
            for window in windows:
                rows.append({"input_ids": window, **meta})
        return rows

    def _tokenize_parallel(self) -> Optional[List[Dict]]:
        """Parallel tokenization from BED paths; None if not applicable."""
        if not self.universe:
            return None
        items = self._path_items()
        if items is None:
            return None

        from functools import partial
        from multiprocessing import Pool

        worker = partial(
            _tokenize_path_worker,
            universe=self.universe,
            context_size=self.context_size,
            max_windows=self.max_windows_per_file,
            drop_unk=self.drop_unk,
        )
        # deterministic per-item seed derived from the base seed + index
        tasks = [(i, path, meta, self.seed + i) for i, (path, meta) in enumerate(items)]

        rows: List[Dict] = []
        with Pool(processes=self.num_proc) as pool:
            for windows, meta in pool.imap_unordered(worker, tasks, chunksize=8):
                for window in windows:
                    rows.append({"input_ids": window, **meta})
        return rows

    def _path_items(self):
        """Return ``[(path, meta), ...]`` if every item is path-backed, else None."""
        path_items_fn = getattr(self.source, "path_items", None)
        if callable(path_items_fn):
            return path_items_fn()
        items = []
        for region_set, meta in self.source:
            path = getattr(region_set, "path", None) or meta.get("path")
            if not path:
                return None
            items.append((path, meta))
        return items


def _tokenize_path_worker(task, universe, context_size, max_windows, drop_unk):
    """Top-level worker: (idx, bed_path, meta, seed) -> (windows, meta).

    Rebuilds the tokenizer from ``universe`` in each process (gtars tokenizers are
    not picklable across processes).
    """
    idx, path, meta, seed = task
    from gtars.models import RegionSet
    from gtars.tokenizers import Tokenizer

    tokenizer = Tokenizer.from_bed(universe) if os.path.isfile(universe) else Tokenizer(universe)
    windows = tokenize_regionset(
        RegionSet(path),
        tokenizer,
        context_size=context_size,
        max_windows=max_windows,
        rng=random.Random(seed),
        drop_unk=drop_unk,
    )
    return windows, meta


_STREAM_CLS = None


def _streaming_class():
    """Build (once, lazily) the concrete IterableDataset subclass.

    ``torch`` is optional, so the class can't be declared at module top; the
    DataLoader relies on ``isinstance(ds, IterableDataset)``, so streaming must be a
    genuine subclass rather than a duck-typed stand-in.
    """
    global _STREAM_CLS
    if _STREAM_CLS is not None:
        return _STREAM_CLS

    try:
        from torch.utils.data import IterableDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("stream mode requires torch (pip install geniml[ml]).") from exc

    class _StreamingTokenizedRegionDataset(IterableDataset):
        """Tokenizes a source on the fly, yielding ``{"input_ids": [...], **meta}``.

        With ``cache_dir`` set, each item's filtered token ids are cached to a
        ``.gtok`` on first pass so later epochs skip re-tokenization (windowing is
        re-done each epoch -- cheap, and keeps windows fresh). Composes with the HF
        ``Trainer`` and the RTD ``DataCollator``.
        """

        def __init__(
            self,
            source,
            tokenizer,
            context_size=8192,
            max_windows_per_file=10,
            cache_dir=None,
            seed=42,
            drop_unk=True,
        ):
            super().__init__()
            self.source = source
            self.tokenizer = tokenizer
            self.context_size = context_size
            self.max_windows_per_file = max_windows_per_file
            self.cache_dir = cache_dir
            self.seed = seed
            self.drop_unk = drop_unk
            self._epoch = 0
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

        def __iter__(self):
            import torch

            worker_info = torch.utils.data.get_worker_info()
            num_workers = worker_info.num_workers if worker_info else 1
            worker_id = worker_info.id if worker_info else 0

            rng = random.Random(self.seed + self._epoch)
            self._epoch += 1

            for i, (region_set, meta) in enumerate(self.source):
                if i % num_workers != worker_id:
                    continue  # shard across DataLoader workers
                ids = self._cached_ids(region_set, meta)
                if not ids:
                    continue
                if (
                    self.max_windows_per_file is not None
                    and len(ids) > self.max_windows_per_file * self.context_size
                ):
                    continue
                for window in sample_and_remove(ids, self.context_size, rng=rng):
                    yield {"input_ids": window, **meta}

        def _cached_ids(self, region_set, meta):
            item_id = meta.get("id") or getattr(region_set, "identifier", None)
            gtok_path = (
                os.path.join(self.cache_dir, f"{item_id}.ctx.gtok")
                if (self.cache_dir and item_id)
                else None
            )
            if gtok_path and os.path.exists(gtok_path):
                from gtars.utils import read_tokens_from_gtok

                return list(read_tokens_from_gtok(gtok_path))

            ids = self.tokenizer(region_set)["input_ids"]
            if self.drop_unk:
                unk_id = getattr(self.tokenizer, "unk_token_id", None)
                if unk_id is not None:
                    ids = [i for i in ids if i != unk_id]

            if gtok_path and ids:
                from gtars.utils import write_tokens_to_gtok

                write_tokens_to_gtok(gtok_path, ids)
            return ids

    _STREAM_CLS = _StreamingTokenizedRegionDataset
    return _STREAM_CLS


def StreamingTokenizedRegionDataset(*args, **kwargs):
    """Construct a streaming tokenized dataset (a ``torch`` ``IterableDataset``).

    A factory rather than a class so the torch dependency stays lazy; the returned
    instance is a genuine ``IterableDataset`` subclass, so the HF ``Trainer`` /
    ``DataLoader`` treat it as iterable-style.
    """
    return _streaming_class()(*args, **kwargs)
