import os

import numpy as np
import pytest
from gitk.region2vec.main import Region2VecExModel
from gitk.search.backends import HNSWBackend, QdrantBackend
from gitk.text2bednn.text2bednn import Text2BEDSearchInterface, Vec2VecFNN
from gitk.text2bednn.utils import (build_regionset_info_list,  # data_split,
                                   prepare_vectors_for_database,
                                   region_info_list_to_vectors)
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split


@pytest.fixture
def metadata_path():
    """
    :return: the path to the metadata file (sorted)
    """
    return "./data/hg38_metadata_sample_sorted.tab"


@pytest.fixture
def bed_folder():
    """
    :return: the path to the folder where bed files are stored
    """
    return "./data/hg38_sample"


@pytest.fixture
def r2v_hf_repo():
    """
    :return: the huggingface repo of Region2VecExModel
    """
    return "databio/r2v-ChIP-atlas"


@pytest.fixture
def r2v_model(r2v_hf_repo):
    """
    :param r2v_hf_repo:
    :return: the Region2VecExModel
    """
    return Region2VecExModel(r2v_hf_repo)


@pytest.fixture
def st_hf_repo():
    """
    :return: the huggingface repo of SentenceTransformer
    """
    return "sentence-transformers/all-MiniLM-L12-v2"


@pytest.fixture
def st_model(st_hf_repo):
    """
    :param st_hf_repo:
    :return: the SentenceTransformer
    """
    return SentenceTransformer(st_hf_repo)


@pytest.fixture
def local_model_path():
    """
    :return: path to save the Vec2VecFNN model, will be deleted after testing
    """
    # return "./testing.keras"
    return "./testing_local_model.h5"


@pytest.fixture
def testing_input():
    """
    :return: a random generated np.ndarray,
    with same dimension as a sentence embedding vector of SentenceTransformer
    """
    np.random.seed(100)
    return np.random.random((384,))


@pytest.fixture
def collection():
    """
    collection name for qdrant client storage
    """

    return "hg38_sample"


@pytest.fixture
def query_term():
    """
    :return: a query string
    """
    return "human, kidney, blood"


@pytest.fixture
def k():
    """
    :return: number of nearest neighbor to search
    """
    return 5


@pytest.fixture
def local_idx_path():
    """
    :return: local file path to save hnsw index,
    will be deleted after testing
    """

    return "./testing_idx.bin"


def test_data_nn_search_interface(
    bed_folder,
    metadata_path,
    r2v_model,
    st_model,
    local_model_path,
    testing_input,
    collection,
    query_term,
    k,
    local_idx_path,
):
    # construct a list of RegionSetInfo
    ri_list = build_regionset_info_list(bed_folder, metadata_path, r2v_model, st_model)
    assert len(ri_list) == len(os.listdir(bed_folder))

    # split the RegionSetInfo list to training, validating, and testing set
    # train_list, test_list = train_test_split(ri_list, test_size=0.15)
    train_list, validate_list = train_test_split(ri_list, test_size=0.2)
    train_X, train_Y = region_info_list_to_vectors(train_list)
    validate_X, validate_Y = region_info_list_to_vectors(validate_list)
    assert isinstance(train_X, np.ndarray)
    assert isinstance(train_Y, np.ndarray)
    assert train_X.shape[1] == 384
    assert train_Y.shape[1] == 100
    assert train_X[0].shape == (384,)
    assert train_Y[0].shape == (100,)

    # fit the Vec2VecFNN model
    v2vnn = Vec2VecFNN()
    v2vnn.train(train_X, train_Y, validating_data=(validate_X, validate_Y), num_epochs=50)

    # save the model to local file
    v2vnn.save(local_model_path, save_format="h5")

    # load pretrained file
    new_e2nn = Vec2VecFNN()
    new_e2nn.load_local_pretrained(local_model_path)

    # testing if the loaded model is same as previously saved model
    map_vec_1 = v2vnn.embedding_to_embedding(testing_input)
    map_vec_2 = new_e2nn.embedding_to_embedding(testing_input)
    map_vec_2 = new_e2nn.embedding_to_embedding(testing_input)
    assert np.array_equal(map_vec_1, map_vec_2)
    # remove locally saved model
    os.remove(local_model_path)

    # train the model without validate data
    X, Y = region_info_list_to_vectors(ri_list)
    v2vnn_no_val = Vec2VecFNN()
    v2vnn_no_val.train(X, Y, num_epochs=50)

    # loading data to search backend
    embeddings, labels = prepare_vectors_for_database(ri_list)
    qd_search_backend = QdrantBackend(collection=collection)
    qd_search_backend.load(embeddings, labels)

    # construct a search interface
    db_interface = Text2BEDSearchInterface(st_model, v2vnn, qd_search_backend)
    db_search_result = db_interface.nl_vec_search(query_term, k)
    for i in range(len(db_search_result)):
        assert isinstance(db_search_result[i], dict)

    # construct a search interface with file backend
    hnsw_backend = HNSWBackend(local_index_path=local_idx_path)
    hnsw_backend.load(embeddings, labels)
    file_interface = Text2BEDSearchInterface(st_model, v2vnn, hnsw_backend)

    file_search_result = file_interface.nl_vec_search(query_term, k)
    for i in range(len(file_search_result)):
        assert isinstance(file_search_result[i], dict)

    # remove local hnsw index
    os.remove(local_idx_path)
