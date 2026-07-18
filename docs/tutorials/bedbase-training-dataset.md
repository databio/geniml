# Build a tokenized training dataset from bedbase

`geniml.dataset` turns a set of region sets into a tokenized training dataset for
region models (atacformer and any other tokenized-region model). It separates two
concerns:

- a **source** — *where* the region sets come from (bedbase, a local file list, a
  `BedSet`, ...), and
- a source-agnostic **dataset** — tokenize + window + batch.

Adding a new data origin is a small source, not a new dataset. Bedbase is the first
concrete source: it turns a *live* bedbase selection into a tokenized dataset.

## The pieces

| Object | Role |
| --- | --- |
| `RegionSetSource` | Protocol: iterate `(RegionSet, meta)` tuples. |
| `BedbaseSource` | Yields region sets from a bedbase selection via `BBClient`. |
| `FileListSource` | Yields region sets from a list/text-file of local BED paths. |
| `BedSetSource` | Wraps a `geniml.io.BedSet` (or `BBClient.load_bedset` output). |
| `select_bedbase_samples` | Query the bedbase API → a dated `SampleSelection` manifest. |
| `TokenizedRegionDataset` | Tokenize+window any source into a training dataset. |

## Bedbase → tokenized dataset

```python
from geniml.atacformer import TrainingTokenizer
from geniml.dataset import (
    select_bedbase_samples,
    BedbaseSource,
    TokenizedRegionDataset,
)

# 1. Select samples from the live bedbase API (replaces a hand-curated CSV).
#    The selection is a dated, serializable manifest — provenance for "all of bedbase".
selection = select_bedbase_samples(genome="hg38", qc="good", limit=500)

# 2. Wrap the selection as a source. Region sets are read via BBClient — from the
#    local bbcache mirror if present, else downloaded and cached.
source = BedbaseSource(selection)

# 3. Tokenize + window into a training dataset against a universe.
tokenizer = TrainingTokenizer("path/to/universe.bed")
ds = TokenizedRegionDataset(
    source,
    tokenizer,
    context_size=8192,
    mode="materialize",           # tokenize once into an on-disk parquet (cached)
    cache_dir="/scratch/bb_cache",
    universe="path/to/universe.bed",  # lets num_proc>1 rebuild the tokenizer in workers
    num_proc=16,
).build()                          # -> a datasets.Dataset ready for a HF Trainer
```

Each row is `{"input_ids": [...], **metadata}`, where the metadata columns
(`description`, `species_name`, `cell_type`, `cell_line`, `tissue`, `assay`,
`antibody`, `target`, `treatment`, ...) are carried through for conditioning and
analysis. Drop them before training if your collator doesn't want them (or rely on
the HF `Trainer`'s `remove_unused_columns`).

### Materialize vs stream

- **`mode="materialize"`** (default) tokenizes the whole selection once into an
  on-disk `datasets.Dataset`. The cache is keyed on *(selection, universe, context
  size)*, so re-running the same selection reloads the parquet instead of
  re-tokenizing. Best for multi-epoch training.
- **`mode="stream"`** returns a `torch` `IterableDataset` that tokenizes on the fly,
  with an optional `.gtok` cache so epoch 2+ skips re-tokenization. Best for "all of
  bedbase, always current, no giant parquet".

```python
stream_ds = TokenizedRegionDataset(
    source, tokenizer, context_size=8192, mode="stream", cache_dir="/scratch/gtok"
).build()
```

## Other sources, same dataset

A local file list (what region2vec's `BEDDataset` did, on the shared interface):

```python
from geniml.dataset import FileListSource, TokenizedRegionDataset

source = FileListSource("bedfiles.txt")   # or FileListSource(["a.bed", "b.bed"])
ds = TokenizedRegionDataset(source, tokenizer).build()
```

An existing `BedSet` (including `BBClient.load_bedset` output):

```python
from geniml.bbclient import BBClient
from geniml.dataset import BedSetSource, TokenizedRegionDataset

bedset = BBClient().load_bedset("your-bedset-id")
ds = TokenizedRegionDataset(BedSetSource(bedset), tokenizer).build()
```

## Reproducibility

Serialize the selection so a run is dated and rebuildable:

```python
selection.to_json("bedbase_selection.json")
# later:
from geniml.dataset import SampleSelection
selection = SampleSelection.from_json("bedbase_selection.json")
```

`geniml.dataset.write_provenance(...)` writes a `dataset_provenance.json` (selection
hash, universe, context size, snapshot date) alongside a trained model, so an "all of
bedbase" run records exactly which bedbase snapshot produced it.
