"""Bedbase as a data source.

The first concrete :class:`~geniml.dataset.RegionSetSource`: turn a *live* bedbase
selection into region sets. This replaces the lost, hand-curated
``all_hg38_good_meta.csv`` -- the sample list is now resolved from the bedbase API
at selection time and serialized as a dated, reproducible manifest.

Two pieces:

- :func:`select_bedbase_samples` -- query the bedbase API for beds of a given genome
  (with a light QC filter) and return :class:`SampleRef`s carrying the metadata
  columns the training pipeline conditions on.
- :class:`BedbaseSource` -- wrap a selection and yield ``(RegionSet, meta)`` by
  reading each bed through :class:`geniml.bbclient.BBClient` (from the local bbcache
  mirror when present, else download+cache). It never queries the bbcache SQLite
  directly.

On QC: the bedbase ``/v1/bed/list`` endpoint exposes ``genome`` and
``bed_compliance`` filters but no explicit "QC-good" flag, so ``qc="good"`` here
means: a real, processed bed for the requested genome (``is_universe`` false) that
meets ``min_compliance``. A stricter region-count filter is available via
``min_regions``/``max_regions`` but requires a per-bed metadata fetch, so it is
opt-in.
"""

import os
from dataclasses import dataclass, asdict, field
from logging import getLogger
from typing import Dict, Iterator, List, Optional

import requests

from .source import RegionSetItem

_LOGGER = getLogger("geniml.dataset")

# Defined locally (not imported from geniml.bbclient.const) so selecting samples
# does not pull in the bbclient package's heavy s3/zarr import chain. Kept in sync
# with geniml.bbclient.const.DEFAULT_BEDBASE_API.
DEFAULT_BEDBASE_API = os.getenv("BEDBASE_API") or "https://api.bedbase.org"

#: bedbase metadata columns preserved on each sample (parity with the frozen parquet).
METADATA_COLUMNS = (
    "description",
    "species_name",
    "cell_type",
    "cell_line",
    "tissue",
    "assay",
    "antibody",
    "target",
    "treatment",
)

#: bed_compliance strings, weakest to strongest, for the ``min_compliance`` filter.
_COMPLIANCE_ORDER = ["bed2+0", "bed3+0", "bed4+0", "bed5+0", "bed6+0"]


@dataclass
class SampleRef:
    """A single selected bedbase sample: its id plus the metadata columns."""

    id: str
    name: Optional[str] = None
    genome: Optional[str] = None
    number_of_regions: Optional[int] = None
    meta: Dict = field(default_factory=dict)

    def as_meta(self) -> Dict:
        """Flatten to a metadata dict suitable for a dataset row."""
        out = {"id": self.id, "name": self.name, "genome": self.genome}
        out.update(self.meta)
        return out


@dataclass
class SampleSelection:
    """A dated, serializable manifest of a bedbase selection (provenance).

    Serialize with :meth:`to_json` so an "all of bedbase" run is reproducible: the
    exact ids, the genome, the QC filter, the API, and the snapshot date.
    """

    genome: str
    qc: str
    bedbase_api: str
    snapshot_date: Optional[str]
    samples: List[SampleRef]

    def ids(self) -> List[str]:
        return [s.id for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def to_json(self, path: str) -> str:
        import json

        payload = {
            "genome": self.genome,
            "qc": self.qc,
            "bedbase_api": self.bedbase_api,
            "snapshot_date": self.snapshot_date,
            "samples": [asdict(s) for s in self.samples],
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path

    @classmethod
    def from_json(cls, path: str) -> "SampleSelection":
        import json

        with open(path, "r") as fh:
            payload = json.load(fh)
        samples = [SampleRef(**s) for s in payload["samples"]]
        return cls(
            genome=payload["genome"],
            qc=payload["qc"],
            bedbase_api=payload["bedbase_api"],
            snapshot_date=payload.get("snapshot_date"),
            samples=samples,
        )


def _compliance_ok(value: Optional[str], minimum: str) -> bool:
    if not minimum:
        return True
    try:
        return _COMPLIANCE_ORDER.index(value) >= _COMPLIANCE_ORDER.index(minimum)
    except ValueError:
        # unknown compliance string -> keep it (don't silently drop unfamiliar beds)
        return True


def _record_to_sampleref(record: Dict) -> SampleRef:
    annotation = record.get("annotation") or {}
    meta = {col: annotation.get(col) for col in METADATA_COLUMNS}
    # bedbase calls the organism 'organism'; the training pipeline expects species_name
    if meta.get("species_name") is None:
        meta["species_name"] = annotation.get("organism")
    return SampleRef(
        id=record["id"],
        name=record.get("name"),
        genome=record.get("genome_alias"),
        meta=meta,
    )


def select_bedbase_samples(
    genome: str = "hg38",
    qc: str = "good",
    limit: Optional[int] = None,
    bedbase_api: str = DEFAULT_BEDBASE_API,
    min_compliance: str = "bed3+0",
    min_regions: Optional[int] = None,
    max_regions: Optional[int] = None,
    page_size: int = 1000,
    snapshot_date: Optional[str] = None,
) -> SampleSelection:
    """Select bedbase samples for a genome, returning a dated selection manifest.

    Args:
        genome: genome alias to filter on (e.g. ``"hg38"``).
        qc: QC policy label recorded in the manifest. ``"good"`` applies the
            ``is_universe``/``min_compliance`` filter; ``"all"`` keeps everything
            for the genome.
        limit: cap the number of selected samples (``None`` = all).
        bedbase_api: bedbase API base URL.
        min_compliance: minimum ``bed_compliance`` to keep (see ``_COMPLIANCE_ORDER``).
        min_regions: if set, drop beds with fewer regions (requires a per-bed
            metadata fetch -- slower).
        max_regions: if set, drop beds with more regions (requires a per-bed fetch).
        page_size: API page size for pagination.
        snapshot_date: ISO date string stamped into the manifest for provenance. If
            ``None``, caller should stamp it (kept out of here so the function is
            deterministic/testable).

    Returns:
        SampleSelection: the selection manifest.
    """
    strict = qc != "all"
    need_stats = min_regions is not None or max_regions is not None

    samples: List[SampleRef] = []
    offset = 0
    session = requests.Session()
    while True:
        url = f"{bedbase_api}/v1/bed/list"
        params = {"genome": genome, "limit": page_size, "offset": offset}
        resp = session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            break

        for record in results:
            if strict:
                if record.get("is_universe"):
                    continue
                if not _compliance_ok(record.get("bed_compliance"), min_compliance):
                    continue
            ref = _record_to_sampleref(record)
            if need_stats:
                n = _fetch_region_count(session, bedbase_api, ref.id)
                ref.number_of_regions = n
                if min_regions is not None and (n is None or n < min_regions):
                    continue
                if max_regions is not None and (n is None or n > max_regions):
                    continue
            samples.append(ref)
            if limit is not None and len(samples) >= limit:
                break

        if limit is not None and len(samples) >= limit:
            break
        offset += page_size
        if offset >= payload.get("count", 0):
            break

    _LOGGER.info("Selected %d bedbase samples for genome=%s (qc=%s)", len(samples), genome, qc)
    return SampleSelection(
        genome=genome,
        qc=qc,
        bedbase_api=bedbase_api,
        snapshot_date=snapshot_date,
        samples=samples,
    )


def _fetch_region_count(session, bedbase_api: str, bed_id: str) -> Optional[int]:
    try:
        resp = session.get(
            f"{bedbase_api}/v1/bed/{bed_id}/metadata", params={"full": "true"}, timeout=60
        )
        resp.raise_for_status()
        stats = resp.json().get("stats") or {}
        n = stats.get("number_of_regions")
        return int(n) if n is not None else None
    except Exception:  # pragma: no cover - network best-effort
        return None


class BedbaseSource:
    """A :class:`~geniml.dataset.RegionSetSource` over a bedbase selection.

    Yields ``(RegionSet, meta)`` by loading each selected bed through a
    :class:`~geniml.bbclient.BBClient` -- from the local bbcache mirror if present,
    otherwise downloading and caching it. Reads only via ``BBClient.load_bed``; it
    never touches the bbcache SQLite directly.
    """

    def __init__(
        self,
        selection: SampleSelection,
        bbclient=None,
        skip_errors: bool = True,
    ):
        """Initialize a BedbaseSource.

        Args:
            selection: a :class:`SampleSelection` (from :func:`select_bedbase_samples`
                or loaded from a manifest).
            bbclient: a configured ``geniml.bbclient.BBClient``; a default one is
                created if omitted (imported lazily, since it pulls in s3/zarr deps).
            skip_errors: if True, samples that fail to load are logged and skipped
                rather than aborting the whole iteration.
        """
        if bbclient is None:
            from ..bbclient.bbclient import BBClient

            bbclient = BBClient()
        self.selection = selection
        self.bbclient = bbclient
        self.skip_errors = skip_errors

    def __len__(self) -> int:
        return len(self.selection)

    def __iter__(self) -> Iterator[RegionSetItem]:
        for ref in self.selection.samples:
            try:
                region_set = self.bbclient.load_bed(ref.id)
            except Exception as exc:  # network / cache miss
                if self.skip_errors:
                    _LOGGER.warning("Skipping bed %s: %s", ref.id, exc)
                    continue
                raise
            yield region_set, ref.as_meta()

    def manifest(self) -> List[str]:
        """Stable list of selected bed ids for cache keying/provenance."""
        return self.selection.ids()

    def path_items(self):
        """Return ``[(bed_path, meta), ...]``, loading (downloading) beds as needed.

        Used by the dataset's parallel materialize path, which tokenizes from BED
        paths in worker processes.
        """
        out = []
        for ref in self.selection.samples:
            try:
                rs = self.bbclient.load_bed(ref.id)
                out.append((rs.path, ref.as_meta()))
            except Exception as exc:
                if self.skip_errors:
                    _LOGGER.warning("Skipping bed %s: %s", ref.id, exc)
                    continue
                raise
        return out
