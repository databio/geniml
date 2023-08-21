import pytest
from gitk.search.dbbackend import QdrantBackend
from gitk.search.filebackend import HNSWBackend
import numpy as np
from qdrant_client.models import VectorParams, Distance
import os
import random


@pytest.fixture
def bed_folder():
    """
    folder where testing bed files are stored
    """

    return "./data/hg38_sample"


@pytest.fixture
def embeddings(filenames):
    """
    mock embedding vectors for testing
    """

    np.random.seed(100)
    return np.random.random((len(filenames), 100))


@pytest.fixture
def filenames(bed_folder):
    """
    list of bed file names
    """

    return os.listdir(bed_folder)


@pytest.fixture
def metadata():
    """
    mock metadata for testing
    """

    return "This is a mock metadata, just for testing."


@pytest.fixture
def labels(filenames, metadata):
    """
    mock list of label dictionaries for testing
    """

    output_list = []
    for name in filenames:
        output_list.append({"name": name,
                            "metadata": metadata})
    return output_list


@pytest.fixture
def config():
    """
    config of Qdrant client
    """

    return VectorParams(size=100, distance=Distance.COSINE)


@pytest.fixture
def collection():
    """
    collection name for qdrant client storage
    """

    return "hg38_sample"


@pytest.fixture
def key(filenames):
    """
    list of randomly sampled ids
    """

    random.seed(100)
    return random.sample(range(len(filenames)), 5)


@pytest.fixture
def local_idx_path():
    """
    local file path to save hnsw index
    """

    return "./testing_idx.bin"


def test_QdrantBackend(filenames, embeddings,
                       labels, config,
                       collection, key):

    # build connection
    qd_search_backend = QdrantBackend(config, collection)
    # load data
    qd_search_backend.load(embeddings, labels)
    # test searching
    search_result = qd_search_backend.search(np.random.random(100, ), 5)
    assert isinstance(search_result, list)
    assert isinstance(search_result[0].id, int)
    assert isinstance(search_result[0].payload, dict)
    assert len(qd_search_backend) == len(filenames)

    # test information retrieval
    retrieval_results = qd_search_backend.retrieve_info(key, True)
    assert isinstance(retrieval_results, dict)
    for id_ in key:
        assert isinstance(retrieval_results[id_], dict)
        assert isinstance(retrieval_results[id_]["embedding"], list)
        assert len(retrieval_results[id_]["embedding"]) == 100
        assert isinstance(retrieval_results[id_]["name"], str)
        assert isinstance(retrieval_results[id_]["metadata"], str)

        client_retrieval = qd_search_backend.qd_client.retrieve(collection,
                                                                [id_],
                                                                with_vectors=True)

        matching_vec = client_retrieval[0].vector
        assert retrieval_results[id_]["embedding"] == matching_vec


def test_HNSWBackend(filenames, embeddings, labels,
                     local_idx_path, key):

    # init backend
    hnswb = HNSWBackend(local_index_path=local_idx_path)
    num_upload = len(filenames)

    # batches to load
    labels_1 = labels[:num_upload // 2]
    labels_2 = labels[num_upload // 2:]

    embeddings_1 = embeddings[:num_upload // 2]
    embeddings_2 = embeddings[num_upload // 2:]

    # load first batch
    hnswb.load(embeddings_1, labels_1)
    assert (len(hnswb) == num_upload // 2)

    # load second batch
    hnswb.load(embeddings_2, labels_2)
    assert (len(hnswb) == num_upload)

    # test searching with one vector (np.ndarray with shape (dim,))
    labels, ids, distances = hnswb.search(np.random.random(100, ), 5)
    assert (ids.dtype == np.uint64)
    assert (ids.shape == (5,))
    assert (distances.shape == (5,))
    assert (len(labels) == 5)
    assert (isinstance(labels[0], dict))

    # test searching with multiple vectors (np.ndarray with shape (n, dim))
    labels, ids, distances = hnswb.search(np.random.random((7, 100)), 5)
    assert (ids.dtype == np.uint64)
    assert (ids.shape == (7, 5))
    assert (distances.shape == (7, 5))
    assert (len(labels) == 7)
    assert (len(labels[0]) == 5)
    assert (isinstance(labels[0][0], dict))

    # test information retrieval / get items
    retrieval_results = hnswb.retrieve_info(key, True)
    for id_ in key:
        assert isinstance(retrieval_results[id_], dict)
        assert isinstance(retrieval_results[id_]["embedding"], list)
        assert len(retrieval_results[id_]["embedding"]) == 100
        assert isinstance(retrieval_results[id_]["name"], str)
        assert isinstance(retrieval_results[id_]["metadata"], str)

        matching_vec = hnswb.idx.get_items([id_])[0]
        assert retrieval_results[id_]["embedding"] == matching_vec

    # remove local file of saved index
    os.remove(local_idx_path)
