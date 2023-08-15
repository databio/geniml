import pytest
import scanpy as sc
import os
from gitk.text2bednn.text2bednn import build_BedMetadataSet_from_files, TextToBedNN, TextToBedNNSearchInterface
from gitk.region2vec.main import Region2Vec, Region2VecExModel
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client.models import VectorParams, Distance
from gitk.search.search import QdrantBackend


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


@pytest.fixture
def query_term():
    return "human, kidney, blood"


@pytest.fixture
def k():
    return 5


def test_data_nn_search(bed_folder, metadata_path,
                        r2v_hf_repo, query_term, k):

    r2v_model = Region2VecExModel(r2v_hf_repo)
    # st_model = SentenceTransformer(st_hf_repo)
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    bme_set = build_BedMetadataSet_from_files(bed_folder, metadata_path, r2v_model, st_model)
    bed_count = len(os.listdir(bed_folder))
    assert bme_set is not None
    assert len(bme_set) == bed_count
    train_X, train_Y = bme_set.generate_data("training")
    assert (isinstance(train_X, np.ndarray))
    assert (isinstance(train_Y, np.ndarray))
    assert (train_X.shape[1] == 384)
    assert (train_Y.shape[1] == 100)
    assert (train_X[0].shape == (384,))
    assert (train_Y[0].shape == (100,))

    t2bnn = TextToBedNN(None, "sentence-transformers/all-MiniLM-L12-v2")
    t2bnn.train(bme_set, epochs=50)

    config = VectorParams(size=100, distance=Distance.COSINE)
    collection = "hg38_sample"

    embeddings, labels = bme_set.to_qd_upload()

    for i in range(bed_count):
        assert np.array_equal(bme_set.tolist[i].region_set_embedding,
                              embeddings[i])
        assert bme_set.tolist[i].file_name == labels[i]

    qd_search_backend = QdrantBackend(config, collection)
    qd_search_backend.load(embeddings, labels)

    t2bnn_interface = TextToBedNNSearchInterface(t2bnn, qd_search_backend)
    search_results = t2bnn_interface.nlsearch(query_term, k)
    print("search resuts:")
    for result in search_results:
        print(result.payload["label"])