import pytest
import scanpy as sc


from geniml.io.io import Region, RegionSet
from geniml.tokenization.main import InMemTokenizer, ITTokenizer


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


def test_make_in_mem_tokenizer_from_file(universe_bed_file: str):
    t = InMemTokenizer(universe_bed_file)
    assert len(t.universe) == 2_378
    assert t is not None


def test_make_in_mem_tokenizer_from_region_set(universe_bed_file: str):
    u = RegionSet(universe_bed_file)
    t = InMemTokenizer(u)
    assert t is not None


def test_tokenize_bed_file(universe_bed_file: str):
    """
    Use the in memory tokenizer to tokenize a bed file.

    The bed file contains 13 regions, 10 of which are in the universe. None
    of them should be the original regions in the file.
    """
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in the bed file to test
    with open(bed_file, "r") as f:
        lines = f.readlines()
        regions = []
        for line in lines:
            chr, start, stop = line.strip().split("\t")
            regions.append(Region(chr, int(start), int(stop)))

    tokens = t.tokenize(bed_file, return_all=True)

    # ensure that the tokens are unqiue from the original regions
    assert len(set(tokens).intersection(set(regions))) == 0
    assert len([t for t in tokens if t is not None]) == 3


def test_tokenize_list_of_regions(universe_bed_file: str):
    """
    Use the in memory tokenizer to tokenize a list of regions.

    The bed file contains 13 regions, 10 of which are in the universe. None
    of them should be the original regions in the file.
    """
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in each and cast as a region
    with open(bed_file, "r") as f:
        lines = f.readlines()
        regions = []
        for line in lines:
            chr, start, stop = line.strip().split("\t")
            regions.append(Region(chr, int(start), int(stop)))

    tokens = t.tokenize(regions, return_all=True)

    # ensure that the tokens are unqiue from the original regions
    assert len(set(tokens).intersection(set(regions))) == 0
    assert len([t for t in tokens if t is not None]) == 3


def test_tokenize_training_data(universe_bed_file: str):
    region_sets = [RegionSet("tests/data/to_tokenize.bed")] * 10
    t = InMemTokenizer(universe_bed_file)

    region_sets_tokenized = [t.tokenize(rs) for rs in region_sets]
    region_sets_ids = [t.convert_tokens_to_ids(rs) for rs in region_sets_tokenized]

    assert len(region_sets_tokenized) == len(region_sets)
    assert len(region_sets_ids) == len(region_sets)


def test_convert_to_ids(universe_bed_file: str):
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in each and cast as a region
    with open(bed_file, "r") as f:
        lines = f.readlines()
        regions = []
        for line in lines:
            chr, start, stop = line.strip().split("\t")
            regions.append(Region(chr, int(start), int(stop)))

    tokens = t.tokenize(regions)
    ids = t.convert_tokens_to_ids(tokens)

    assert len(ids) == len(tokens)
    assert all(isinstance(i, int) or i is None for i in ids)


def test_tokenize_anndata(universe_bed_file: str, pbmc_data: sc.AnnData):
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    tokens = t.tokenize(pbmc_data, return_all=True)

    # returns list of regions for each cell
    assert len(tokens) == pbmc_data.shape[0]


@pytest.mark.skip(reason="This test is not working yet")
def test_tokenize_anndata_backed(universe_bed_file: str, pbmc_data_backed: sc.AnnData):
    t = InMemTokenizer(universe_bed_file)
    assert t is not None

    tokens = t.tokenize(pbmc_data_backed)

    assert len(tokens) == pbmc_data_backed.shape[0]


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


def test_gtokenize_to_bit_vector(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)
    assert t is not None

    # tokenize a bed file
    bed_file = "tests/data/to_tokenize.bed"

    # read in the bed file to test
    rs = RegionSet(bed_file)

    # tokenize
    tokens = t.tokenize(rs)
    bit_vector = tokens.bit_vector

    assert all(isinstance(b, bool) for b in bit_vector)
    assert len(bit_vector) == 2_379


def test_ittokenizer_get_padding(universe_bed_file: str):
    t = ITTokenizer(universe_bed_file)
    padding = t.padding_token()

    assert padding.id == 2379
