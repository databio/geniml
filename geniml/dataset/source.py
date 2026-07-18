"""Region-set sources.

A *source* is anything that yields :class:`gtars.models.RegionSet` objects plus
per-item metadata. It is the pluggable half of the dataset: where the region sets
come from. A source does **no** tokenizing -- it only produces region sets. Adding
a new data origin (local files, bedbase, anndata, ...) means writing a small source,
not a new dataset.

The interface is deliberately minimal (see :class:`RegionSetSource`):

- ``__iter__`` yields ``(RegionSet, meta)`` tuples (:data:`RegionSetItem`).
- ``__len__`` returns the number of items, or ``None`` if unknown/streaming.
- ``manifest`` (optional) returns a stable list of item ids so the dataset can
  key an on-disk cache on the exact selection.
"""

import os
from logging import getLogger
from typing import Dict, Iterator, List, Optional, Tuple, Union
from typing_extensions import Protocol, runtime_checkable

from gtars.models import RegionSet

_LOGGER = getLogger("geniml.dataset")

#: A single item produced by a source: a region set plus its metadata dict.
RegionSetItem = Tuple[RegionSet, Dict]


@runtime_checkable
class RegionSetSource(Protocol):
    """Protocol for anything that yields region sets plus metadata.

    A source is the pluggable input to :class:`~geniml.dataset.TokenizedRegionDataset`.
    Any object that is iterable over ``(RegionSet, meta)`` tuples satisfies it --
    ``geniml.io.BedSet`` already very nearly does.
    """

    def __iter__(self) -> Iterator[RegionSetItem]:  # pragma: no cover - protocol
        ...

    def __len__(self) -> Optional[int]:  # pragma: no cover - protocol
        ...


class BedSetSource:
    """Wrap a :class:`geniml.io.BedSet` (or ``BBClient.load_bedset`` output) as a source.

    ``BedSet`` already holds a list of ``RegionSet``s and is iterable, so this
    adapter is thin: it just attaches metadata (the region set's identifier and
    path) to each item. Use it to feed an existing bedset -- including one returned
    by :meth:`geniml.bbclient.BBClient.load_bedset` -- straight into the dataset.
    """

    def __init__(self, bedset, meta: Optional[Dict] = None):
        """Initialize a BedSetSource.

        Args:
            bedset: a ``geniml.io.BedSet`` (anything iterable over RegionSets with
                a ``__len__``).
            meta: optional metadata merged into every item's metadata dict (e.g.
                the bedset identifier), useful for provenance/conditioning.
        """
        self.bedset = bedset
        self.meta = dict(meta or {})
        identifier = getattr(bedset, "identifier", None)
        if identifier and "bedset_id" not in self.meta:
            self.meta["bedset_id"] = identifier

    def __len__(self) -> Optional[int]:
        try:
            return len(self.bedset)
        except TypeError:
            return None

    def __iter__(self) -> Iterator[RegionSetItem]:
        for region_set in self.bedset:
            meta = dict(self.meta)
            meta.setdefault("id", getattr(region_set, "identifier", None))
            meta.setdefault("path", getattr(region_set, "path", None))
            yield region_set, meta

    def manifest(self) -> Optional[List[str]]:
        """Stable list of item ids (region-set identifiers) for cache keying."""
        try:
            return [getattr(rs, "identifier", None) or getattr(rs, "path") for rs in self.bedset]
        except Exception:  # pragma: no cover - defensive
            return None


class FileListSource:
    """A source over a list of local BED file paths.

    This is exactly what region2vec's ``BEDDataset`` did (a text file listing local
    BED paths), expressed on the shared source interface. Accepts either a path to a
    text file (one BED path per line) or an in-memory list of paths.
    """

    def __init__(self, files: Union[str, os.PathLike, List[str]], root: Optional[str] = None):
        """Initialize a FileListSource.

        Args:
            files: path to a text file listing one BED path per line, OR a list of
                BED file paths.
            root: optional directory prepended to each (relative) path.
        """
        if isinstance(files, (str, os.PathLike)) and os.path.isfile(files):
            with open(files, "r") as fh:
                paths = [line.strip() for line in fh if line.strip()]
        elif isinstance(files, (list, tuple)):
            paths = [str(p) for p in files]
        else:
            raise ValueError(
                "`files` must be a path to a text file listing BED paths, or a list of paths."
            )
        if root:
            paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in paths]
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[RegionSetItem]:
        for path in self.paths:
            rs = RegionSet(path)
            meta = {
                "id": getattr(rs, "identifier", None),
                "path": path,
                "name": os.path.basename(path),
            }
            yield rs, meta

    def manifest(self) -> List[str]:
        """Stable list of item ids (the sorted file paths)."""
        return sorted(self.paths)
