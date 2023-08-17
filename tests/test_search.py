import pytest
from gitk.search.search import QdrantBackend, HNSWBackend
import numpy as np
from qdrant_client.models import VectorParams, Distance
import os



@pytest.fixture
def bed_folder():
    return "./data/hg38_sample"


@pytest.fixture
def filenames(bed_folder):
    return os.listdir(bed_folder)


@pytest.fixture
def embeddings(filenames):
    np.random.seed(100)
    return np.random.random((len(filenames), 100))


@pytest.fixture
def local_idx_path():
    return "./testing_idx.bin"


def test_QdrantBackend(filenames, embeddings):
    # r2v_model = Region2VecExModel(r2v_hf_repo)
    config = VectorParams(size=100, distance=Distance.COSINE)
    collection = "hg38_sample"

    qd_search_backend = QdrantBackend(config, collection)
    qd_search_backend.load(embeddings, filenames)
    search_result = qd_search_backend.search(np.random.random(100,), 5)
    assert isinstance(search_result, list)
    assert isinstance(search_result[0].id, int)
    assert isinstance(search_result[0].payload, dict)
    assert len(qd_search_backend) == len(filenames)


def test_HNSWBackend(filenames, embeddings, local_idx_path):

    hnswb = HNSWBackend(local_index_path=local_idx_path)
    num_upload = len(filenames)

    filenames_1 = filenames[:num_upload//2]
    filenames_2 = filenames[num_upload//2:]

    embeddings_1 = embeddings[:num_upload//2]
    embeddings_2 = embeddings[num_upload//2:]

    hnswb.load(embeddings_1, filenames_1)

    assert(len(hnswb) == num_upload//2)

    hnswb.load(embeddings_2, filenames_2)

    assert (len(hnswb) == num_upload)

    ids, distances = hnswb.search(np.random.random(100,), 5)

    assert(ids.dtype == np.uint64)

    os.remove(local_idx_path)
