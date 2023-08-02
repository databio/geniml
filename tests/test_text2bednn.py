import pytest
import scanpy as sc
import os
from gitk.text2bednn.text2bednn import build_BedMetadataSet_from_files
from gitk.region2vec.main import Region2Vec, Region2VecExModel
from sentence_transformers import SentenceTransformer
import numpy as np


@pytest.fixture
def metadata_path():
    return "./data/hg38_metadata_sample_sorted.tab"


@pytest.fixture
def bed_folder():
    return "./data/hg38_sample"


@pytest.fixture
def r2v_hf_repo():
    return "databio/r2v-ChIP-atlas"


@pytest.fixture
def st_hf_repo():
    return "sentence-transformers/all-MiniLM-L12-v2"


def test_BedMetadataEmbeddingSet(bed_folder, metadata_path, r2v_hf_repo):
    r2v_model = Region2VecExModel(r2v_hf_repo)
    # st_model = SentenceTransformer(st_hf_repo)
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    bme_set = build_BedMetadataSet_from_files(bed_folder, metadata_path, r2v_model, st_model)
    assert bme_set is not None
    assert len(bme_set) == 20
    train_X, train_Y = bme_set.generate_data("training")
    assert(isinstance(train_X, np.ndarray))
    assert (isinstance(train_Y, np.ndarray))
    assert(train_X.shape[1] == 384)
    assert(train_Y.shape[1] == 100)
    assert(train_X[0].shape == (384, ))
    assert (train_Y[0].shape == (100,))
