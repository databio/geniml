import os
import random
from typing import Dict, List

import numpy as np
import pytest

from geniml.io import RegionSet
from geniml.region2vec.main import Region2VecExModel
from geniml.search import BED2BEDSearchInterface, BED2Vec, Text2BEDSearchInterface, Text2Vec
from geniml.search.backends import HNSWBackend, QdrantBackend
from geniml.search.backends.filebackend import DEP_HNSWLIB

DATA_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data"
)


@pytest.fixture
def bed_folder():
    """
    folder where testing bed files are stored
    """

    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "data",
        "hg38_sample",
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


# @pytest.fixture
@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    # temporal index folder
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def temp_idx_path(temp_data_dir):
    # temporal index path
    return temp_data_dir / "testing_idx.bin"


@pytest.fixture(scope="module")
def hnswb(temp_idx_path):
    # init backend
    return HNSWBackend(local_index_path=str(temp_idx_path))


@pytest.fixture
def bed_folder():
    """
    :return: the path to the folder where testing bed files are stored
    """
    return os.path.join(DATA_FOLDER_PATH, "hg38_sample")


@pytest.fixture
def r2v_hf_repo():
    """
    :return: the huggingface repo of region2vec model
    """
    return "databio/r2v-encode-hg38"


@pytest.fixture
def v2v_hf_repo():
    """
    Returns: the huggingface repo of vec2vec model
    """
    return "databio/v2v-sentencetransformers-encode"


@pytest.fixture
def collection():
    """
    Returns: collection name for qdrant client storage
    """

    return "hg38_sample"


@pytest.fixture
def query_term():
    """
    Returns: a query string
    """
    return "human, kidney, blood"


@pytest.fixture
def nl_embed_repo():
    """
    Returns: text embedding model for search backend
    """
    return "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def query_bed():
    """
    Returns: path to a BED file in testing data
    """
    return "./data/s1_a.bed"


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
        assert isinstance(result, dict)  # only target pairs
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


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
def test_HNSWBackend_load(filenames, embeddings, labels, hnswb, ids):
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
    # pytestconfig.cache.set('shared_backend', hnswb)


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
# @pytest.mark.dependency(depends=["test_HNSWBackend_load"])
def test_HNSWBackend_search(filenames, hnswb, ids):
    def search_result_check(dict_list: List[Dict], backend: HNSWBackend, with_dist: bool = False):
        """
        repeated test of the output of search / retrieve_info function of HNSWBackend to check if the result matches the content in index

        :param dict_list: the result, which is supposed to be a list of dictionary
        :param backend: the HNSWBackend to be tested
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
            assert isinstance(result["vector"], np.ndarray)
            # assert result["vector"] == index.get_items([result["id"]])[0]
            assert (
                result["vector"] == index.get_items([result["id"]], return_type="numpy")[0]
            ).all()
            for num in result["vector"]:
                assert isinstance(num, np.float32)

    # hnswb = pytestconfig.cache.get('shared_backend', None)
    assert len(hnswb) == len(filenames)
    # test searching with one vector (np.ndarray with shape (dim,))
    query_vec = np.random.random(
        100,
    )
    single_vec_search = hnswb.search(
        query_vec,
        3,
    )

    single_vec_search_offset = hnswb.search(
        query_vec,
        3,
        offset=2,
    )

    for j in range(len(single_vec_search)):
        assert single_vec_search_offset[j]["id"] == single_vec_search[j]["id"]
        assert single_vec_search_offset[j]["distance"] == single_vec_search[j]["distance"]
        assert (
            single_vec_search_offset[j]["payload"]["metadata"]
            == single_vec_search[j]["payload"]["metadata"]
        )
    search_result_check(single_vec_search, hnswb, True)
    search_result_check(single_vec_search_offset, hnswb, True)

    # test searching with multiple vectors (np.ndarray with shape (n, dim))
    multiple_vecs_search = hnswb.search(np.random.random((7, 100)), 5)
    assert isinstance(multiple_vecs_search, list)
    assert len(multiple_vecs_search) == 7
    for i in range(len(multiple_vecs_search)):
        search_result_check(multiple_vecs_search[i], hnswb, True)

    # test information retrieval / get items
    retrieval_results = hnswb.retrieve_info(ids, True)
    search_result_check(retrieval_results, hnswb, False)


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
# @pytest.mark.dependency(depends=["test_HNSWBackend_load"])
def test_HNSWBackend_save(filenames, hnswb, embeddings, temp_idx_path, temp_data_dir):
    # test saving from local
    new_hnswb = HNSWBackend(local_index_path=str(temp_idx_path), payloads=hnswb.payloads)
    assert new_hnswb.idx.max_elements == embeddings.shape[0]

    for i in range(embeddings.shape[0]):
        old_result = hnswb.idx.get_items([i], return_type="numpy")
        new_result = new_hnswb.idx.get_items([i], return_type="numpy")
        assert (old_result == new_result).all()

    # test a bug:
    new_idx_path = temp_data_dir / "new_idx.bin"
    empty_hnswb = HNSWBackend(local_index_path=str(new_idx_path))
    assert len(empty_hnswb.payloads) == 0


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
def test_bed2vec(r2v_hf_repo, bed_folder):
    bed2vec = BED2Vec(r2v_hf_repo)
    for bed_name in os.listdir(bed_folder):
        assert bed2vec.forward(os.path.join(bed_folder, bed_name)).shape == (100,)


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
def test_text2vec(nl_embed_repo, v2v_hf_repo):
    text2vec = Text2Vec(nl_embed_repo, v2v_hf_repo)
    assert text2vec.forward("Hematopoietic cells").shape == (100,)


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
@pytest.mark.skipif(
    "not config.getoption('--qdrant')",
    reason="Only run when --qdrant is given",
)
def test_text2bed_search_interface(
    bed_folder,
    r2v_hf_repo,
    nl_embed_repo,
    v2v_hf_repo,
    collection,
    query_term,
    tmp_path_factory,
):
    r2v_model = Region2VecExModel(r2v_hf_repo)
    query_dict = {"Mock1": [2, 3], "Mock2": [1], "Mock3": [2, 4, 5], "Mock4": [0]}
    vecs = []
    payloads = []

    for bed_name in os.listdir(bed_folder):
        rs = RegionSet(os.path.join(bed_folder, bed_name))
        # bed_vecs.append(r2v.encode(bed_path))
        region_embeddings = r2v_model.encode(rs)
        bed_file_embedding = np.mean(region_embeddings, axis=0)
        vecs.append(bed_file_embedding)
        file_payload = {"name": bed_name}
        payloads.append(file_payload)

    vecs = np.array(vecs)

    qd_search_backend = QdrantBackend(collection=collection)
    qd_search_backend.load(vectors=vecs, payloads=payloads)
    # #
    text2vec = Text2Vec(nl_embed_repo, v2v_hf_repo)
    # #
    # # construct a search interface
    db_interface = Text2BEDSearchInterface(qd_search_backend, text2vec)
    db_search_result = db_interface.query_search(query_term, 5, offset=0)
    for i in range(len(db_search_result)):
        assert isinstance(db_search_result[i], dict)

    eval_results = db_interface.eval(query_dict)
    assert eval_results["Mean Average Precision"] > 0
    assert eval_results["Mean AUC-ROC"] > 0
    assert eval_results["Average R-Precision"] > 0

    # delete testing collection
    db_interface.backend.qd_client.delete_collection(collection_name=collection)

    # construct a search interface with file backend
    temp_data_dir = tmp_path_factory.mktemp("data")
    temp_idx_path = temp_data_dir / "testing_idx.bin"
    hnsw_backend = HNSWBackend(local_index_path=str(temp_idx_path))
    hnsw_backend.load(vectors=vecs, payloads=payloads)
    file_interface = Text2BEDSearchInterface(hnsw_backend, text2vec)

    file_search_result = file_interface.query_search(query_term, 5)
    for i in range(len(file_search_result)):
        assert isinstance(file_search_result[i], dict)

    # test evaluation
    eval_results = file_interface.eval(query_dict)
    assert eval_results["Mean Average Precision"] > 0
    assert eval_results["Mean AUC-ROC"] > 0
    assert eval_results["Average R-Precision"] > 0


@pytest.mark.skipif(
    "not config.getoption('--huggingface')",
    reason="Only run when --huggingface is given",
)
@pytest.mark.skipif(
    "not config.getoption('--qdrant')",
    reason="Only run when --qdrant is given",
)
def test_bed2bed_search_interface(
    bed_folder,
    r2v_hf_repo,
    collection,
    query_bed,
    tmp_path_factory,
):
    r2v_model = Region2VecExModel(r2v_hf_repo)
    vecs = []
    payloads = []

    for bed_name in os.listdir(bed_folder):
        rs = RegionSet(os.path.join(bed_folder, bed_name))
        region_embeddings = r2v_model.encode(rs)
        bed_file_embedding = np.mean(region_embeddings, axis=0)
        vecs.append(bed_file_embedding)
        file_payload = {"name": bed_name}
        payloads.append(file_payload)

    vecs = np.array(vecs)

    qd_search_backend = QdrantBackend(collection=collection)
    qd_search_backend.load(vectors=vecs, payloads=payloads)

    bed2vec = BED2Vec(r2v_hf_repo)

    # # construct a search interface
    db_interface = BED2BEDSearchInterface(qd_search_backend, bed2vec)
    db_search_result = db_interface.query_search(query_bed, 5, offset=0)
    for i in range(len(db_search_result)):
        assert isinstance(db_search_result[i], dict)

    # delete testing collection
    db_interface.backend.qd_client.delete_collection(collection_name=collection)

    # construct a search interface with file backend
    temp_data_dir = tmp_path_factory.mktemp("data")
    temp_idx_path = temp_data_dir / "testing_idx.bin"
    hnsw_backend = HNSWBackend(local_index_path=str(temp_idx_path))
    hnsw_backend.load(vectors=vecs, payloads=payloads)
    file_interface = Text2BEDSearchInterface(hnsw_backend, bed2vec)

    file_search_result = file_interface.query_search(query_bed, 5)
    for i in range(len(file_search_result)):
        assert isinstance(file_search_result[i], dict)
