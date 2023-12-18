import pytest
import scanpy as sc

from geniml.io.io import RegionSet
from geniml.tokenization.main import ITTokenizer


@pytest.fixture
def pbmc_data():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def pbmc_data_backed():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad", backed="r")


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.uniq.bed"


def test_create_universe(universe_bed_file: str):
    u = RegionSet(universe_bed_file)
    assert u is not None
    assert len(u) == 2_378


def test_make_ittokenizer(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)
    assert t is not None


def test_ittokenize_region_set(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in the bed file to test
    rs = RegionSet(bed_file)

    # tokenize
    tokens = t.tokenize(rs)
    region_tokens = tokens.regions

    # filter out UNK tokens
    assert len(region_tokens) == 14  # one of the regions gets split into two tokens
    region_tokens = [t for t in region_tokens if t.chr != "chrUNK"]

    # count tokens
    assert len([t for t in region_tokens if t is not None]) == 3


def test_yield_tokens(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in the bed file to test
    rs = RegionSet(bed_file)

    # tokenize
    tokens = t.tokenize(rs)
    region_tokens = tokens.regions

    # filter out UNK tokens
    assert len(region_tokens) == 14  # one of the regions gets split into two tokens
    region_tokens = [t for t in region_tokens if t.chr != "chrUNK"]

    # count tokens
    assert len([t for t in region_tokens if t is not None]) == 3

    # yield tokens
    for token in tokens:
        assert token is not None
        assert isinstance(token.id, int)


def test_tokenize_anndata(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)

    # tokenize anndata
    adata = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")

    # tokenize
    tokens = t.tokenize(adata)

    # count tokens
    assert len(tokens) == 20


def test_ittokenizer_get_padding(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)
    padding = t.padding_token()

    assert padding.id == 2379
