import pytest

from geniml.io import Region
from geniml.preprocessing import RegionIDifier
from geniml.preprocessing.utils import wordify_region
from geniml.preprocessing.schemas import EncodedRegions


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

    assert r.word_to_id("[PAD]") == 0
    assert r.word_to_id("[UNK]") == 1
    assert r.word_to_id("[CLS]") == 2
    assert r.word_to_id("[SEP]") == 3
    assert r.word_to_id("[MASK]") == 4
    assert r.id_to_word(0) == "[PAD]"
    assert r.id_to_word(1) == "[UNK]"
    assert r.id_to_word(2) == "[CLS]"
    assert r.id_to_word(3) == "[SEP]"
    assert r.id_to_word(4) == "[MASK]"


def test_generate_region_ids(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garuanteed to be in the vocab
    regions = [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
    ]

    regions_as_tokens = [wordify_region(r) for r in regions]

    ids1 = idifier.convert_regions_to_ids(regions)
    ids2 = idifier.convert_tokens_to_ids(regions_as_tokens)

    assert ids1 == [5, 6, 7]
    assert ids2 == [5, 6, 7]


def test_convert_ids_to_tokens(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garuanteed to be in the vocab
    ids = [5, 6, 7]

    regions = idifier.convert_ids_to_tokens(ids)
    assert regions == [
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
    ]


def test_convert_ids_to_regions(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garuanteed to be in the vocab
    ids = [5, 6, 7]

    regions = idifier.convert_ids_to_regions(ids)
    assert regions == [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
    ]


def test_attention_mask_generation(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # these are the regions in the universe bed file, garuanteed to be in the vocab
    ids = [5, 6, 7]  # see above tests for what these ids correspond to

    # attention should be given to all tokens except the padding token, there
    # are no padding tokens in this example, so the attention mask should be all 1s
    attention_mask = idifier.generate_attention_mask_from_ids(ids)
    assert attention_mask == [1, 1, 1]

    attention_mask = idifier.generate_attention_mask_from_tokens(
        idifier.convert_ids_to_tokens(ids)
    )
    assert attention_mask == [1, 1, 1]

    # now add a padding token
    ids = [5, 6, 0]
    attention_mask = idifier.generate_attention_mask_from_ids(ids)
    assert attention_mask == [1, 1, 0]

    attention_mask = idifier.generate_attention_mask_from_tokens(
        idifier.convert_ids_to_tokens(ids)
    )
    assert attention_mask == [1, 1, 0]


def test_unknown_tokens_and_ids(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    this_token_doesnt_exist = "chr1_1_2"
    id = idifier.word_to_id(this_token_doesnt_exist)

    assert id == 1

    this_id_doesnt_exist = 1000000000
    token = idifier.id_to_word(this_id_doesnt_exist)
    assert token == "[UNK]"


def test_padding(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    ids = [5, 6, 0]

    padded_ids = idifier.pad_ids(ids, 10)
    assert padded_ids == [5, 6, 0, 0, 0, 0, 0, 0, 0, 0]

    tokens = [
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
    ]

    padded_tokens = idifier.pad_tokens(tokens, 10)
    assert padded_tokens == [
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
        "[PAD]",
    ]


def test_truncation(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    ids = [5, 6, 0, 5, 6, 5, 6, 5, 6, 5, 6]

    truncated_ids = idifier.truncate_ids(ids, 2)
    assert truncated_ids == [5, 6, 0]

    tokens = [
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
    ]

    truncated_tokens = idifier.truncate_tokens(tokens, 2)
    assert truncated_tokens == [
        "chr1_63403166_63403785",
        "chr10_20783474_20784387",
        "chr1_121484861_121485361",
    ]


def test_encode(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    # test basic functionality
    # these are the regions in the universe bed file, garuanteed to be in the vocab
    regions = [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
    ]

    tokens1 = idifier.tokenize(regions)
    tokens2 = idifier(regions)  # check __call__ method

    assert isinstance(tokens1, EncodedRegions)
    assert isinstance(tokens2, EncodedRegions)
    assert tokens1.ids == tokens2.ids
    assert tokens1.attention_mask == tokens2.attention_mask

    regions1 = [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
    ]
    regions2 = [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
    ]

    tokens = idifier([regions1, regions2])

    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert isinstance(tokens[0], EncodedRegions)
    assert isinstance(tokens[1], EncodedRegions)

    # check for padding on the first batch
    assert tokens[0].ids == [
        2,
        5,
        6,
        7,
        5,
        6,
        0,
        0,
        0,
        0,
        0,
    ]  # 0 is the padding token id, 2 is the cls token id
    assert tokens[0].attention_mask == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert tokens[1].ids == [
        2,
        5,
        6,
        7,
        5,
        6,
        5,
        6,
        7,
        5,
        6,
    ]  # 0 is the padding token id, 2 is the cls token id
    assert tokens[1].attention_mask == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # now test the truncation functionality
    idifier = RegionIDifier(transformer_vocab_file, max_length=6)

    tokens = idifier([regions1, regions2])

    assert isinstance(tokens, list)
    assert len(tokens) == 2
    assert isinstance(tokens[0], EncodedRegions)
    assert isinstance(tokens[1], EncodedRegions)

    # ensure length is 6 (+ 1 for cls token)
    assert len(tokens[0].ids) == 7
    assert len(tokens[1].ids) == 7

    # ensure things were padded correctly
    assert tokens[0].ids == [
        2,
        5,
        6,
        7,
        5,
        6,
        0,
    ]  # 0 is the padding token id, 2 is the cls token id
    assert tokens[0].attention_mask == [1, 1, 1, 1, 1, 1, 0]
    assert tokens[1].ids == [
        2,
        5,
        6,
        7,
        5,
        6,
        5,
    ]  # 0 is the padding token id, 2 is the cls token id
    assert tokens[1].attention_mask == [1, 1, 1, 1, 1, 1, 1]


def test_masked_langauge_modeling(
    transformer_vocab_file: str,
):
    idifier = RegionIDifier(transformer_vocab_file)

    regions = [
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
        Region("chr1", 121484861, 121485361),
        Region("chr1", 63403166, 63403785),
        Region("chr10", 20783474, 20784387),
    ]

    tokens = idifier.tokenize(regions)
    tokens_masked = idifier.mask_tokens(tokens.ids)

    assert len(tokens.ids) == len(tokens_masked.ids)
