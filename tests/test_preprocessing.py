import pytest

from gitk.io import Region
from gitk.preprocessing import RegionIDifier


@pytest.fixture
def universe_bed_file():
    return "tests/data/universe.bed"


@pytest.fixture
def transformer_vocab_file():
    return "tests/data/transformer_vocab.txt"


def test_load_regionidifier(transformer_vocab_file: str):
    r = RegionIDifier(
        transformer_vocab_file,
    )

    assert r.word_to_id["[PAD]"] == 0
    assert r.word_to_id["[UNK]"] == 1
    assert r.word_to_id["[CLS]"] == 2
    assert r.word_to_id["[SEP]"] == 3
    assert r.word_to_id["[MASK]"] == 4
    assert r.id_to_word[0] == "[PAD]"
    assert r.id_to_word[1] == "[UNK]"
    assert r.id_to_word[2] == "[CLS]"
    assert r.id_to_word[3] == "[SEP]"
    assert r.id_to_word[4] == "[MASK]"


def test_generate_region_ids(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garanteed to be in the vocab
    regions = [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
    ]

    ids = idifier.convert_regions_to_ids(regions)
    assert ids == [5, 6, 7]


def test_convert_ids_to_tokens(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garanteed to be in the vocab
    ids = [5, 6, 7]

    regions = idifier.convert_ids_to_regions(ids)
    assert regions == [
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
    ]


def test_convert_ids_to_regions(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garanteed to be in the vocab
    ids = [5, 6, 7]

    regions = idifier.convert_ids_to_regions(ids)
    assert regions == [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
    ]
