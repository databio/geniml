"""Source-agnostic tokenized datasets for region-based models.

Separates **source** (where region sets come from) from **dataset** (tokenize +
window + batch), built on the shared ``RegionSet``/``BedSet`` abstraction. A new
data origin is a small :class:`RegionSetSource`, not a new dataset -- so bedbase,
local file lists, and bedsets all feed the same
:class:`TokenizedRegionDataset`.

Typical use (bedbase -> tokenized training dataset)::

    from geniml.atacformer import TrainingTokenizer
    from geniml.dataset import (
        select_bedbase_samples, BedbaseSource, TokenizedRegionDataset,
    )

    selection = select_bedbase_samples(genome="hg38", qc="good", limit=500)
    source = BedbaseSource(selection)
    tokenizer = TrainingTokenizer("path/to/universe.bed")
    ds = TokenizedRegionDataset(source, tokenizer, context_size=8192,
                                cache_dir="/scratch/bb_cache").build()

The heavy deps (``datasets``, ``torch``) are imported lazily inside the dataset's
methods, so importing this package works on a base geniml install; only building a
dataset requires ``geniml[ml]``.
"""

from .source import (
    BedSetSource,
    FileListSource,
    RegionSetItem,
    RegionSetSource,
)
from .tokenize import sample_and_remove, tokenize_regionset
from .bedbase import (
    METADATA_COLUMNS,
    BedbaseSource,
    SampleRef,
    SampleSelection,
    select_bedbase_samples,
)
from .dataset import StreamingTokenizedRegionDataset, TokenizedRegionDataset
from .manifest import cache_key, source_manifest, universe_signature, write_provenance

__all__ = [
    # source interface
    "RegionSetSource",
    "RegionSetItem",
    "BedSetSource",
    "FileListSource",
    # tokenization core
    "tokenize_regionset",
    "sample_and_remove",
    # bedbase source + selection
    "select_bedbase_samples",
    "BedbaseSource",
    "SampleRef",
    "SampleSelection",
    "METADATA_COLUMNS",
    # dataset
    "TokenizedRegionDataset",
    "StreamingTokenizedRegionDataset",
    # manifest / provenance
    "cache_key",
    "source_manifest",
    "universe_signature",
    "write_provenance",
]
