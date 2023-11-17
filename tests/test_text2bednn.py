import os

import numpy as np
import pytest
from fastembed.embedding import FlagEmbedding
from sklearn.model_selection import train_test_split
from torchsummary import summary

from geniml.region2vec.main import Region2VecExModel
from geniml.search.backends import HNSWBackend, QdrantBackend
from geniml.text2bednn.text2bednn import Text2BEDSearchInterface, Vec2VecFNN
from geniml.text2bednn.utils import (
    build_regionset_info_list_from_files,
    build_regionset_info_list_from_PEP,
    prepare_vectors_for_database,
    region_info_list_to_vectors,
    vectors_from_backend,
)
from geniml.tokenization.main import ITTokenizer


@pytest.fixture
def metadata_path():
    """
    :return: the path to the metadata file (sorted)
    """
    return "./data/testing_hg38_metadata_sorted.tab"


@pytest.fixture
def bed_folder():
    """
    :return: the path to the folder where bed files are stored
    """
    return "./data/hg38_sample"


@pytest.fixture
def universe_path():
    """
    :return: the universe file for tokenizer
    """
    return "./data/universe.bed"


@pytest.fixture
def tokenizer(universe_path):
    """
    :return: a tokenizer
    """
    return ITTokenizer(universe_path)


@pytest.fixture
def r2v_model(bed_folder, tokenizer):
    """
    :return: a Region2VecExModel that is trained within very short of time
    """
    r2v_model = Region2VecExModel(model_path=None, tokenizer=tokenizer, min_count=1)
    r2v_model.train([f"{bed_folder}/{name}" for name in os.listdir(bed_folder)], epochs=15)

    return r2v_model


@pytest.fixture
def r2v_hf_repo():
    """
    :return: the huggingface repo of Region2VecExModel
    """
    return "databio/r2v-ChIP-atlas-hg38-v2"


@pytest.fixture
def r2v_hf_model(r2v_hf_repo):
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
def nl_embed(st_hf_repo):
    """
    :param st_hf_repo:
    :return: the SentenceTransformer
    """
    return FlagEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def local_model_path():
    """
    :return: path to save the Vec2VecFNN model, will be deleted after testing
    """
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
def testing_input_biogpt():
    """
    :return: a random generated np.ndarray,
    with same dimension as a sentence embedding vector of SentenceTransformer
    """
    np.random.seed(100)
    return np.random.random((1024,))


@pytest.fixture
def yaml_path():
    """
    :return: path to the yaml file of testing PEP
    """
    return "./data/testing_hg38.yaml"


@pytest.fixture
def col_names():
    """
    :return: the columns that are needed in the testing PEP's csv for metadata
    """
    return [
        "tissue",
        "cell_line",
        "tissue_lineage",
        "tissue_description",
        "diagnosis",
        "sample_name",
        "antibody",
    ]


def test_reading_data(bed_folder, metadata_path, yaml_path, col_names, r2v_model, nl_embed):
    """
    Test reading data from files and PEP
    """
    ri_list_PEP = build_regionset_info_list_from_PEP(
        yaml_path,
        col_names,
        r2v_model,
        nl_embed,
    )
    X, Y = region_info_list_to_vectors(ri_list_PEP)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.shape[1] == 384
    assert Y.shape[1] == 100

    ri_list_file = build_regionset_info_list_from_files(
        bed_folder, metadata_path, r2v_model, nl_embed
    )
    X, Y = region_info_list_to_vectors(ri_list_file)
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.shape[1] == 384
    assert Y.shape[1] == 100


def test_torch_running(tmp_path_factory):
    training_X = np.random.random((90, 1024))
    training_Y = np.random.random((90, 100))

    validating_X = np.random.random((10, 1024))
    validating_Y = np.random.random((10, 100))

    best_embed_folder = tmp_path_factory.mktemp("best_embed")

    v2v_torch1 = Vec2VecFNN()

    v2v_torch1.train(
        training_X,
        training_Y,
        (validating_X, validating_Y),
        save_best=False,
        folder_path=best_embed_folder,
        early_stop=True,
        patience=0.1,
        loss_func="cosine_embedding_loss",
        num_epochs=100,
        batch_size=16,
        num_units=[512, 256],
        num_extra_hidden_layers=1,
    )
    v2v_torch1.plot_training_hist(best_embed_folder)
    v2v_torch1.export(best_embed_folder, "v2v_best_epoch.pt")
    summary(v2v_torch1.model)

    v2v_torch2 = Vec2VecFNN()
    v2v_torch2.load_from_disk(
        os.path.join(best_embed_folder, "v2v_best_epoch.pt"),
        os.path.join(best_embed_folder, "config.yaml"),
    )

    input_vecs = np.random.random((5, 1024))

    output1 = v2v_torch1.embedding_to_embedding(input_vecs)
    output2 = v2v_torch2.embedding_to_embedding(input_vecs)

    assert np.array_equal(output1, output2)

    # train the model without validate data
    v2v_torch2.train(training_X, training_Y, num_epochs=10)


@pytest.mark.skipif(
    "not config.getoption('--r2vhf')",
    reason="Only run when --r2vhf is given",
)
@pytest.mark.skipif(
    "not config.getoption('--qdrant')",
    reason="Only run when --qdrant is given",
)
def test_data_nn_search_interface(
    bed_folder,
    metadata_path,
    r2v_hf_model,
    nl_embed,
    local_model_path,
    testing_input,
    collection,
    query_term,
    k,
    tmp_path_factory,
):
    def test_vector_from_backend(search_backend, nl_embed):
        """
        repeated test of vectors_from_backend
        """
        # get the vectors
        X, Y = vectors_from_backend(search_backend, nl_embed)
        assert X.shape == (len(search_backend), 384)
        assert Y.shape == (len(search_backend), 100)

        # see if the vectors match the storage from backend
        for i in range(len(search_backend)):
            retrieval = search_backend.retrieve_info(i, with_vec=True)
            assert np.array_equal(np.array(retrieval["vector"]), Y[i])
            nl_embedding = next(nl_embed.embed(retrieval["payload"]["metadata"]))
            assert np.array_equal(nl_embedding, X[i])

    # construct a list of RegionSetInfo
    ri_list = build_regionset_info_list_from_files(
        bed_folder, metadata_path, r2v_hf_model, nl_embed
    )
    assert len(ri_list) == len(os.listdir(bed_folder))

    # split the RegionSetInfo list to training, validating, and testing set
    train_list, validate_list = train_test_split(ri_list, test_size=0.2)
    train_X, train_Y = region_info_list_to_vectors(train_list)
    validate_X, validate_Y = region_info_list_to_vectors(validate_list)
    assert isinstance(train_X, np.ndarray)
    assert isinstance(train_Y, np.ndarray)
    assert train_X.shape[1] == 384
    assert train_Y.shape[1] == 100

    # fit the Vec2VecFNN model
    v2vnn = Vec2VecFNN()
    v2vnn.train(train_X, train_Y, validating_data=(validate_X, validate_Y), num_epochs=50)

    # loading data to search backend
    embeddings, labels = prepare_vectors_for_database(ri_list)
    qd_search_backend = QdrantBackend(collection=collection)
    qd_search_backend.load(vectors=embeddings, payloads=labels)

    # construct a search interface
    db_interface = Text2BEDSearchInterface(nl_embed, v2vnn, qd_search_backend)
    db_search_result = db_interface.nl_vec_search(query_term, k, offset=0)
    for i in range(len(db_search_result)):
        assert isinstance(db_search_result[i], dict)
    # test vectors_from_backend
    test_vector_from_backend(db_interface.search_backend, nl_embed)
    # delete testing collection
    db_interface.search_backend.qd_client.delete_collection(collection_name=collection)

    # construct a search interface with file backend
    temp_data_dir = tmp_path_factory.mktemp("data")
    temp_idx_path = temp_data_dir / "testing_idx.bin"
    hnsw_backend = HNSWBackend(local_index_path=str(temp_idx_path))
    hnsw_backend.load(vectors=embeddings, payloads=labels)
    file_interface = Text2BEDSearchInterface(nl_embed, v2vnn, hnsw_backend)

    file_search_result = file_interface.nl_vec_search(query_term, k)
    for i in range(len(file_search_result)):
        assert isinstance(file_search_result[i], dict)

    test_vector_from_backend(file_interface.search_backend, nl_embed)
