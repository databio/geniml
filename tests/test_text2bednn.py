import pytest
# import scanpy as sc
import os
# from gitk.text2bednn.text2bednn import build_BedMetadataSet_from_files, TextToBedNN, TextToBedNNSearchInterface
from gitk.text2bednn.utils import RegionsetInfo, build_RegionsetInfo_list, data_split, RI_list_to_vectors
from gitk.text2bednn.text2bednn import Embed2EmbedNN
from gitk.region2vec.main import Region2Vec, Region2VecExModel
from sentence_transformers import SentenceTransformer
import numpy as np


# from qdrant_client.models import VectorParams, Distance
# from gitk.search.search import QdrantBackend


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
def r2v_model(r2v_hf_repo):
    return Region2VecExModel(r2v_hf_repo)


@pytest.fixture
def st_hf_repo():
    return "sentence-transformers/all-MiniLM-L12-v2"


@pytest.fixture
def st_model(st_hf_repo):
    return SentenceTransformer(st_hf_repo)


@pytest.fixture
def local_model_path():
    return "./testing_keras.h5"


@pytest.fixture
def testing_input():
    np.random.seed(100)
    return np.random.random((384, ))

# @pytest.fixture
# def query_term():
#     return "human, kidney, blood"


# @pytest.fixture
# def k():
#     return 5


def test_RegionsetInfo_list(bed_folder, metadata_path,
                            r2v_model, st_model, local_model_path,
                            testing_input):
    ri_list = build_RegionsetInfo_list(bed_folder, metadata_path,
                                       r2v_model, st_model)
    assert len(ri_list) == len(os.listdir(bed_folder))

    train_list, validate_list, test_list = data_split(ri_list)

    train_X, train_Y = RI_list_to_vectors(train_list)
    validate_X, validate_Y = RI_list_to_vectors(validate_list)

    assert (isinstance(train_X, np.ndarray))
    assert (isinstance(train_Y, np.ndarray))
    assert (train_X.shape[1] == 384)
    assert (train_Y.shape[1] == 100)
    assert (train_X[0].shape == (384,))
    assert (train_Y[0].shape == (100,))

    e2enn = Embed2EmbedNN()
    e2enn.train(train_X, train_Y, validate_X, validate_Y)

    e2enn.export(local_model_path)

    new_e2nn = Embed2EmbedNN()

    new_e2nn.load_local_pretrained(local_model_path)

    map_vec_1 = e2enn.embedding_to_embedding(testing_input)
    map_vec_2 = new_e2nn.embedding_to_embedding(testing_input)

    assert map_vec_1 == map_vec_2


#
# def test_data_nn_search(bed_folder, metadata_path,
#                         r2v_hf_repo, query_term, k):
#
#     r2v_model = Region2VecExModel(r2v_hf_repo)
#     # st_model = SentenceTransformer(st_hf_repo)
#     st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
#     bme_set = build_BedMetadataSet_from_files(bed_folder, metadata_path, r2v_model, st_model)
#     bed_count = len(os.listdir(bed_folder))
#     assert bme_set is not None
#     assert len(bme_set) == bed_count
#     train_X, train_Y = bme_set.generate_data("training")
#     assert (isinstance(train_X, np.ndarray))
#     assert (isinstance(train_Y, np.ndarray))
#     assert (train_X.shape[1] == 384)
#     assert (train_Y.shape[1] == 100)
#     assert (train_X[0].shape == (384,))
#     assert (train_Y[0].shape == (100,))
#
#     t2bnn = TextToBedNN(None, "sentence-transformers/all-MiniLM-L12-v2")
#     t2bnn.train(bme_set, epochs=50)
#
#     config = VectorParams(size=100, distance=Distance.COSINE)
#     collection = "hg38_sample"
#
#     embeddings, labels = bme_set.to_qd_upload()
#
#     for i in range(bed_count):
#         assert np.array_equal(bme_set.tolist[i].region_set_embedding,
#                               embeddings[i])
#         assert bme_set.tolist[i].file_name == labels[i]
#
#     qd_search_backend = QdrantBackend(config, collection)
#     qd_search_backend.load(embeddings, labels)
#
#     t2bnn_interface = TextToBedNNSearchInterface(t2bnn, qd_search_backend)
#     search_results = t2bnn_interface.nlsearch(query_term, k)
#     print("search resuts:")
#     for result in search_results:
#         print(result.payload["label"])
