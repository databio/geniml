import collections
import os
import random
from typing import Dict, List

import numpy as np
import pytest
from geniml.search.backends import HNSWBackend, QdrantBackend
from geniml.search.backends.filebackend import DEP_HNSWLIB


@pytest.fixture
def bed_folder():
    """
    folder where testing bed files are stored
    """

    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data", "hg38_sample"
    )


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
        output_list.append({"name": name, "metadata": metadata})
    return output_list


@pytest.fixture
def collection():
    """
    collection name for qdrant client storage
    """

    return "hg38_sample"


@pytest.fixture
def ids(filenames):
    """
    list of randomly sampled ids
    """

    random.seed(100)
    return random.sample(range(len(filenames)), 5)


@pytest.mark.skipif(
    "not config.getoption('--qdrant')",
    reason="Only run when --qdrant is given",
)
def test_QdrantBackend(filenames, embeddings, labels, collection, ids):
    qd_search_backend = QdrantBackend(collection=collection)
    # load data
    qd_search_backend.load(embeddings, payloads=labels)
    # test searching
    search_results = qd_search_backend.search(
        np.random.random(
            100,
        ),
        5,
        with_payload=True,
        with_vectors=True,
    )
    assert isinstance(search_results, list)
    for result in search_results:
        assert isinstance(result, dict)
        assert isinstance(result["id"], int)
        assert isinstance(result["score"], float)
        assert isinstance(result["vector"], list)
        for i in result["vector"]:
            assert isinstance(i, float)
        assert isinstance(result["payload"], dict)
        assert isinstance(result["payload"]["name"], str)
        assert isinstance(result["payload"]["metadata"], str)
    assert len(qd_search_backend) == len(filenames)

    # test information retrieval
    retrieval_results = qd_search_backend.retrieve_info(ids, True)
    assert isinstance(retrieval_results, list)
    for i in range(len(ids)):
        assert ids[i] == retrieval_results[i]["id"]

        client_retrieval = qd_search_backend.qd_client.retrieve(
            collection, [ids[i]], with_vectors=True
        )

        assert retrieval_results[i]["vector"] == client_retrieval[0].vector
        assert retrieval_results[i]["payload"] == client_retrieval[0].payload
    qd_search_backend.qd_client.delete_collection(qd_search_backend.collection)


# https://github.com/pyjanitor-devs/pyjanitor/issues/115
@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
def test_HNSWBackend(filenames, embeddings, labels, tmp_path_factory, ids):
    def test_hnsw_search_result(
        dict_list: List[Dict], backend: HNSWBackend, with_dist: bool = False
    ):
        """
        repeated test of the output of search / retrieve_info function of HNSWBackend

        :param dict_list: the result, which is supposed to be a list of dictionary
        :param index: the hnswlib.Index of the backend
        :param with_dist: whether distance score is included in the result
        :return:
        """
        index = backend.idx
        assert isinstance(dict_list, list)
        for result in dict_list:
            assert isinstance(result, dict)
            assert isinstance(result["id"], int)
            if with_dist:
                assert isinstance(result["distance"], float)
            assert isinstance(result["payload"], dict)
            assert isinstance(result["vector"], list)
            assert result["vector"] == index.get_items([result["id"]])[0]
            for num in result["vector"]:
                assert isinstance(num, float)

    # temporal index file
    temp_data_dir = tmp_path_factory.mktemp("data")
    temp_idx_path = temp_data_dir / "testing_idx.bin"
    # init backend
    hnswb = HNSWBackend(local_index_path=str(temp_idx_path))
    num_upload = len(filenames)

    # batches to load
    labels_1 = labels[: num_upload // 2]
    labels_2 = labels[num_upload // 2 :]

    embeddings_1 = embeddings[: num_upload // 2]
    embeddings_2 = embeddings[num_upload // 2 :]

    # load first batch
    hnswb.load(embeddings_1, payloads=labels_1)
    assert len(hnswb) == num_upload // 2

    # load second batch
    hnswb.load(embeddings_2, payloads=labels_2)
    assert len(hnswb) == num_upload

    # test searching with one vector (np.ndarray with shape (dim,))
    query_vec = np.random.random(
        100,
    )
    single_vec_search = hnswb.search(
        query_vec,
        3,
        # with_payload=False,
        # with_vectors=False,
    )

    single_vec_search_offset = hnswb.search(
        query_vec,
        3,
        # with_payload=False,
        # with_vectors=False,
        offset=2,
    )

    assert single_vec_search_offset == single_vec_search
    test_hnsw_search_result(single_vec_search, hnswb, True)
    test_hnsw_search_result(single_vec_search_offset, hnswb, True)

    # test searching with multiple vectors (np.ndarray with shape (n, dim))
    multiple_vecs_search = hnswb.search(np.random.random((7, 100)), 5)
    assert isinstance(multiple_vecs_search, list)
    assert len(multiple_vecs_search) == 7
    for i in range(len(multiple_vecs_search)):
        test_hnsw_search_result(multiple_vecs_search[i], hnswb, True)

    # test information retrieval / get items
    retrieval_results = hnswb.retrieve_info(ids, True)
    test_hnsw_search_result(retrieval_results, hnswb, False)
