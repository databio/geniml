import scanpy as sc

from geniml.tokenization.main import ITTokenizer
from geniml.classification.utils import generate_fine_tuning_dataset


def test_generate_finetuning_dataset():
    t = ITTokenizer("tests/data/universe.bed")
    adata = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")

    pos, neg, pos_labels, neg_labels = generate_fine_tuning_dataset(adata, t)

    assert len(pos) == len(neg)
    assert len(pos) == len(pos_labels)
    assert len(neg) == len(neg_labels)
