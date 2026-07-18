"""Tests for geniml.dataset: source interface, windowing, selection, dataset."""

import os
import random

import pytest

from gtars.models import RegionSet
from gtars.tokenizers import Tokenizer

from geniml.dataset import (
    BedSetSource,
    FileListSource,
    RegionSetSource,
    SampleRef,
    SampleSelection,
    TokenizedRegionDataset,
    cache_key,
    sample_and_remove,
    select_bedbase_samples,
    tokenize_regionset,
)
from geniml.dataset.bedbase import _compliance_ok, _record_to_sampleref
from geniml.io import BedSet

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
UNIVERSE = os.path.join(DATA, "universe.bed")
BED1 = os.path.join(DATA, "to_tokenize.bed")
BED2 = os.path.join(DATA, "to_tokenize2.bed")


@pytest.fixture
def tokenizer():
    return Tokenizer.from_bed(UNIVERSE)


# ---------------------------------------------------------------- windowing


def test_sample_and_remove_windows_cover_all_tokens():
    ids = list(range(10))
    windows = sample_and_remove(ids, context_size=4, rng=random.Random(0))
    assert [len(w) for w in windows] == [4, 4, 2]
    # every token appears exactly once across windows
    assert sorted(t for w in windows for t in w) == ids


def test_sample_and_remove_empty_and_bad_context():
    assert sample_and_remove([], 8) == []
    with pytest.raises(ValueError):
        sample_and_remove([1, 2], 0)


def test_sample_and_remove_deterministic_with_seed():
    ids = list(range(20))
    a = sample_and_remove(ids, 5, rng=random.Random(42))
    b = sample_and_remove(ids, 5, rng=random.Random(42))
    assert a == b


def test_tokenize_regionset_filters_unk(tokenizer):
    rs = RegionSet(BED1)
    windows = tokenize_regionset(rs, tokenizer, context_size=8192, max_windows=10)
    unk = tokenizer.unk_token_id
    assert all(unk not in w for w in windows)
    assert sum(len(w) for w in windows) > 0


def test_tokenize_regionset_skips_huge_files(tokenizer):
    rs = RegionSet(BED1)
    # max_windows tiny + context_size 1 forces the skip branch
    windows = tokenize_regionset(rs, tokenizer, context_size=1, max_windows=1)
    assert windows == []


# ------------------------------------------------------------------ sources


def test_filelist_source_from_list_iterates_and_manifest():
    source = FileListSource([BED1, BED2])
    assert isinstance(source, RegionSetSource)
    assert len(source) == 2
    items = list(source)
    assert len(items) == 2
    rs, meta = items[0]
    assert isinstance(rs, RegionSet)
    assert meta["name"] == "to_tokenize.bed"
    assert source.manifest() == sorted([BED1, BED2])


def test_filelist_source_from_textfile(tmp_path):
    listing = tmp_path / "beds.txt"
    listing.write_text(f"{BED1}\n{BED2}\n")
    source = FileListSource(str(listing))
    assert len(source) == 2


def test_bedset_source_wraps_bedset():
    bs = BedSet([BED1, BED2])
    source = BedSetSource(bs)
    assert isinstance(source, RegionSetSource)
    assert len(source) == 2
    items = list(source)
    assert len(items) == 2
    assert all("id" in meta for _, meta in items)


# --------------------------------------------------------- bedbase selection


def test_compliance_filter():
    assert _compliance_ok("bed6+0", "bed3+0")
    assert not _compliance_ok("bed2+0", "bed3+0")
    assert _compliance_ok("weird-format", "bed3+0")  # unknown -> kept


def test_record_to_sampleref_maps_metadata():
    record = {
        "id": "abc",
        "name": "sample1",
        "genome_alias": "hg38",
        "annotation": {
            "organism": "Homo sapiens",
            "cell_type": "Tcell",
            "assay": "ATAC-seq",
        },
    }
    ref = _record_to_sampleref(record)
    assert ref.id == "abc"
    assert ref.genome == "hg38"
    assert ref.meta["cell_type"] == "Tcell"
    assert ref.meta["species_name"] == "Homo sapiens"  # organism -> species_name
    assert "assay" in ref.meta


def test_sample_selection_roundtrip(tmp_path):
    sel = SampleSelection(
        genome="hg38",
        qc="good",
        bedbase_api="https://api.bedbase.org",
        snapshot_date="2026-07-18",
        samples=[SampleRef(id="a", name="A", genome="hg38", meta={"assay": "ATAC-seq"})],
    )
    path = tmp_path / "manifest.json"
    sel.to_json(str(path))
    loaded = SampleSelection.from_json(str(path))
    assert loaded.ids() == ["a"]
    assert loaded.samples[0].meta["assay"] == "ATAC-seq"
    assert loaded.snapshot_date == "2026-07-18"


# ------------------------------------------------------------- cache keying


def test_cache_key_is_stable_and_selection_sensitive(tokenizer):
    s1 = FileListSource([BED1, BED2])
    s2 = FileListSource([BED2, BED1])  # order-independent
    s3 = FileListSource([BED1])
    k1 = cache_key(s1, tokenizer, 8192)
    k2 = cache_key(s2, tokenizer, 8192)
    k3 = cache_key(s3, tokenizer, 8192)
    assert k1 == k2
    assert k1 != k3
    assert cache_key(s1, tokenizer, 4096) != k1  # context size matters


# ------------------------------------------------------- materialize dataset


def test_materialize_dataset_and_cache(tmp_path, tokenizer):
    pytest.importorskip("datasets")
    source = FileListSource([BED1, BED2])
    ds_builder = TokenizedRegionDataset(
        source, tokenizer, context_size=2, cache_dir=str(tmp_path), max_windows_per_file=100
    )
    ds = ds_builder.materialize()
    assert len(ds) > 0
    assert "input_ids" in ds.column_names
    # a parquet cache was written and reloads
    cached = [f for f in os.listdir(tmp_path) if f.endswith(".parquet")]
    assert cached
    ds2 = ds_builder.materialize()
    assert len(ds2) == len(ds)


# ----------------------------------------------------------- network (opt-in)


@pytest.fixture
def bedbase(request):
    if not request.config.getoption("--bedbase"):
        pytest.skip("use --bedbase to run bedbase network tests")


def test_select_bedbase_samples_live(bedbase):
    selection = select_bedbase_samples(genome="hg38", qc="good", limit=5)
    assert len(selection) == 5
    assert all(s.genome == "hg38" for s in selection.samples)
    assert all(s.id for s in selection.samples)
