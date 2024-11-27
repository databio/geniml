import os
import random
from typing import Dict, List

import numpy as np
import pytest
from geniml.io import RegionSet
from geniml.region2vec.main import Region2VecExModel
from geniml.search import BED2BEDSearchInterface, BED2Vec, Text2BEDSearchInterface, Text2Vec
from geniml.search.backends import BiVectorBackend, HNSWBackend, QdrantBackend
from geniml.search.backends.filebackend import DEP_HNSWLIB

DATA_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data"
)

random.seed(100)
np.random.seed(100)


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
def filenames(bed_folder):
    """
    list of bed file names
    """

    return [
        "ENCX3P",
        "ENCN4Z",
        "ENC7VQ",
        "ENCY6R",
        "ENCJ9K",
        "ENCD8T",
        "ENCQ1A",
        "ENCM2F",
        "ENCKMR",
    ]


@pytest.fixture
def metadata():
    """
    mock metadata for testing
    """

    return {
        "ENCX3P": {"biosample": "HEK293", "target": "H3K27ac", "organ": ["kidney", "epithelium"]},
        "ENCN4Z": {"biosample": "HEK293", "target": "CTCF", "organ": ["kidney"]},
        "ENC7VQ": {"biosample": "HEK293", "target": "TBP", "organ": ["kidney", "epithelium"]},
        "ENCY6R": {"biosample": "A549", "target": "H3K27ac", "organ": ["epithelium", "lung"]},
        "ENCJ9K": {"biosample": "A549", "target": "CTCF", "organ": ["lung"]},
        "ENCD8T": {"biosample": "K562", "target": "TBP", "organ": ["blood"]},
        "ENCQ1A": {"biosample": "K562", "target": "H3K27ac", "organ": ["blood"]},
        "ENCM2F": {"biosample": "K562", "target": "CTCF", "organ": ["blood"]},
        "ENCKMR": {"biosample": "apple"},
    }


@pytest.fixture
def annotations():
    return [
        "HEK293",
        "A549",
        "K562",
        "H3K27ac",
        "CTCF",
        "TBP",
        "kidney",
        "epithelium",
        "lung",
        "blood",
        "apple",
    ]


@pytest.fixture
def annotation_matches():
    return {
        "HEK293": [0, 1, 2],
        "A549": [3, 4],
        "K562": [5, 6, 7],
        "H3K27ac": [0, 3, 5],
        "CTCF": [1, 4, 7],
        "TBP": [2, 6],
        "kidney": [0, 2],
        "epithelium": [0, 2, 3],
        "lung": [3, 4],
        "blood": [5, 6, 7],
        "apple": [8],
    }


@pytest.fixture
def uuids():
    return [
        "7bbab414-053d-4c06-9085-d3ca894dc8b8",
        "8b3fa142-8866-4b4c-9df8-7734b4ef9f2a",
        "478c9b96-3b4c-41c3-af56-68e8c39de0a3",
        "971c58a5-c126-433b-887c-4184184cbce6",
        "b28381f8-82ce-4b19-86c0-34a2d368e3b3",
        "d6f1060e-6e14-4faf-8711-f25ed5c6618e",
        "920ef6f6-f821-46f9-9d11-3516119feeec",
        "6ec9d4a4-a481-43dc-81f3-098953c77b0a",
        "ce5345d8-84a1-4c6c-9427-145a3f207805",
    ]


@pytest.fixture
def bed_payloads(filenames, metadata):
    """
    mock list of label dictionaries for testing
    """

    output_list = []
    for name in filenames:
        output_list.append({"name": name, "metadata": metadata[name]})
    return output_list


@pytest.fixture
def metadata_payloads(annotations, annotation_matches):
    """
    mock list of label dictionaries for testing
    """

    output_list = []
    for tag in annotations:
        output_list.append({"text": tag, "matched_files": annotation_matches[tag]})
    return output_list


@pytest.fixture
def bed_collection():
    """
    Returns: bed_collection name for qdrant client storage
    """

    return "hg38_sample"


@pytest.fixture
def metadata_collection():
    """
    Returns: bed_collection name for qdrant client storage
    """

    return "bed_metadata"


@pytest.fixture
def bed_embeddings(filenames):
    """
    mock embedding vectors for testing
    """

    return np.random.random((len(filenames), 100))


@pytest.fixture
def text_embeddings(annotations):
    """
    mock embedding vectors for testing
    """

    return np.random.random((len(annotations), 384))


@pytest.fixture
def int_ids(filenames):
    """
    list of randomly sampled integer_ids
    """
    return random.sample(range(len(filenames)), 3)


@pytest.fixture
def ids(uuids):
    ids_with_hyphen = random.sample(uuids, 3)
    return [uuid.replace("-", "") for uuid in ids_with_hyphen]


# @pytest.fixture
@pytest.fixture(scope="module")
def temp_data_dir(tmp_path_factory):
    # temporal index folder
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def temp_bed_idx_path(temp_data_dir):
    # temporal index path
    return temp_data_dir / "testing_idx.bin"


@pytest.fixture(scope="module")
def temp_metadata_idx_path(temp_data_dir):
    # temporal index path
    return temp_data_dir / "testing_metadata_idx.bin"


@pytest.fixture(scope="module")
def bed_hnswb(temp_bed_idx_path):
    # init backend
    return HNSWBackend(local_index_path=str(temp_bed_idx_path))


@pytest.fixture(scope="module")
def metadata_hnswb(temp_metadata_idx_path):
    # init backend
    return HNSWBackend(local_index_path=str(temp_metadata_idx_path), dim=384)


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


def cosine_similarity(vec1: np.array, vec2: np.array) -> float:
    # Ensure the vectors have shape (100,)
    assert vec1.shape == (100,) and vec2.shape == (100,), "Both vectors must have shape (100,)"

    # Compute the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)

    # Compute the magnitude (L2 norm) of each vector
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    # Compute the cosine similarity
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0  # Avoid division by zero

    cosine_sim = dot_product / (magnitude_vec1 * magnitude_vec2)

    return cosine_sim


@pytest.mark.skipif(
    "not config.getoption('--qdrant')",
    reason="Only run when --qdrant is given",
)
def test_QdrantBackend(filenames, bed_embeddings, bed_payloads, bed_collection, ids, uuids):
    def search_results_test(search_results):
        assert isinstance(search_results, list)
        for result in search_results:
            assert isinstance(result, dict)  # only target pairs
            assert isinstance(result["id"], str)
            assert isinstance(result["score"], float)

            assert isinstance(result["vector"], list)
            for i in result["vector"]:
                assert isinstance(i, float)
            assert isinstance(result["payload"], dict)
            assert isinstance(result["payload"]["name"], str)
            assert isinstance(result["payload"]["metadata"], dict)

    qd_search_backend = QdrantBackend(collection=bed_collection)
    # load data
    qd_search_backend.load(bed_embeddings, payloads=bed_payloads, ids=uuids)
    # test searching
    query_vec = np.random.random(
        100,
    )
    search_results = qd_search_backend.search(
        query_vec,
        5,
        with_payload=True,
        with_vectors=True,
    )

    search_results_test(search_results)

    assert len(qd_search_backend) == len(filenames)

    # test information retrieval
    retrieval_results = qd_search_backend.retrieve_info(ids, True)
    assert len(retrieval_results) == len(ids)
    assert isinstance(retrieval_results, list)
    for i in range(len(ids)):
        assert ids[i] == retrieval_results[i]["id"]

        client_retrieval = qd_search_backend.qd_client.retrieve(
            bed_collection, [ids[i]], with_vectors=True
        )

        assert retrieval_results[i]["vector"] == client_retrieval[0].vector
        assert retrieval_results[i]["payload"] == client_retrieval[0].payload

    # test batch search
    batch_query = np.random.random((6, 100))
    batch_result = qd_search_backend.search(
        batch_query, limit=3, with_payload=True, with_vectors=True
    )
    assert len(batch_result) == 6
    for batch in batch_result:
        search_results_test(batch)

    qd_search_backend.qd_client.delete_collection(qd_search_backend.collection)


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
def test_HNSWBackend_load(filenames, bed_embeddings, bed_payloads, bed_hnswb, ids):
    num_upload = len(filenames)

    # batches to load
    labels_1 = bed_payloads[: num_upload // 2]
    labels_2 = bed_payloads[num_upload // 2 :]
    embeddings_1 = bed_embeddings[: num_upload // 2]
    embeddings_2 = bed_embeddings[num_upload // 2 :]

    # load first batch
    bed_hnswb.load(embeddings_1, payloads=labels_1)
    assert len(bed_hnswb) == num_upload // 2

    # load second batch
    bed_hnswb.load(embeddings_2, payloads=labels_2)
    assert len(bed_hnswb) == num_upload


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
@pytest.mark.dependency(depends=["test_HNSWBackend_load"])
def test_HNSWBackend_search(filenames, bed_hnswb, int_ids):
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

            assert (
                result["vector"] == index.get_items([result["id"]], return_type="numpy")[0]
            ).all()
            for num in result["vector"]:
                assert isinstance(num, np.float32)

    # bed_hnswb = pytestconfig.cache.get('shared_backend', None)
    assert len(bed_hnswb) == len(filenames)
    # test searching with one vector (np.ndarray with shape (dim,))
    query_vec = np.random.random(
        100,
    )
    single_vec_search = bed_hnswb.search(
        query_vec,
        3,
    )

    single_vec_search_offset = bed_hnswb.search(
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
    search_result_check(single_vec_search, bed_hnswb, True)
    search_result_check(single_vec_search_offset, bed_hnswb, True)

    # test searching with multiple vectors (np.ndarray with shape (n, dim))
    multiple_vecs_search = bed_hnswb.search(np.random.random((7, 100)), 5)
    assert isinstance(multiple_vecs_search, list)
    assert len(multiple_vecs_search) == 7
    for i in range(len(multiple_vecs_search)):
        search_result_check(multiple_vecs_search[i], bed_hnswb, True)

    # test information retrieval / get items
    retrieval_results = bed_hnswb.retrieve_info(int_ids, True)
    search_result_check(retrieval_results, bed_hnswb, False)


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
@pytest.mark.dependency(depends=["test_HNSWBackend_load"])
def test_HNSWBackend_save(filenames, bed_hnswb, bed_embeddings, temp_bed_idx_path, temp_data_dir):
    # test saving from local
    new_hnswb = HNSWBackend(local_index_path=str(temp_bed_idx_path), payloads=bed_hnswb.payloads)
    assert new_hnswb.idx.max_elements == bed_embeddings.shape[0]

    for i in range(bed_embeddings.shape[0]):
        old_result = bed_hnswb.idx.get_items([i], return_type="numpy")
        new_result = new_hnswb.idx.get_items([i], return_type="numpy")
        assert (old_result == new_result).all()

    # test a bug:
    new_idx_path = temp_data_dir / "new_idx.bin"
    empty_hnswb = HNSWBackend(local_index_path=str(new_idx_path))
    assert len(empty_hnswb.payloads) == 0


@pytest.mark.skipif(
    DEP_HNSWLIB == False, reason="This test require installation of hnswlib (optional)"
)
@pytest.mark.dependency(depends=["test_HNSWBackend_load"])
@pytest.mark.skipif(
    "not config.getoption('--qdrant')",
    reason="Only run when --qdrant is given",
)
def test_BiVectorBackend(
    bed_hnswb,
    metadata_hnswb,
    bed_collection,
    bed_embeddings,
    bed_payloads,
    metadata_collection,
    text_embeddings,
    metadata_payloads,
):
    def bivec_test(bivec_backend, dist: bool = False, rank: bool = False):
        query_vec = np.random.random(
            384,
        )
        search_results = bivec_backend.search(
            query_vec, 2, with_payload=True, with_vectors=True, distance=dist, rank=rank
        )
        assert isinstance(search_results, list)
        assert len(search_results) == 2
        min_score = 100.0
        max_rank = -1
        for result in search_results:
            assert isinstance(result, dict)  # only target pairs
            assert isinstance(result["id"], int)

            if not rank:
                assert isinstance(result["score"], float)
                assert 0 <= result["score"] <= 1
                assert result["score"] <= min_score
                min_score = result["score"]
            else:
                assert isinstance(result["max_rank"], int)
                assert result["max_rank"] >= max_rank
                max_rank = result["max_rank"]

            assert isinstance(result["vector"], list) or isinstance(result["vector"], np.ndarray)
            if isinstance(result["vector"], list):
                for i in result["vector"]:
                    assert isinstance(i, float)
            assert isinstance(result["payload"], dict)
            assert isinstance(result["payload"]["name"], str)
            assert isinstance(result["payload"]["metadata"], dict)

    # test QdrantBackend
    bed_backend = QdrantBackend(collection=bed_collection)
    # load data
    bed_backend.load(bed_embeddings, payloads=bed_payloads)

    text_backend = QdrantBackend(collection=metadata_collection, dim=384)
    text_backend.load(text_embeddings, payloads=metadata_payloads)

    bivec_qd_backend = BiVectorBackend(text_backend, bed_backend)
    bivec_test(bivec_qd_backend, rank=True)
    bivec_test(bivec_qd_backend, rank=False)
    bivec_qd_backend.metadata_backend.qd_client.delete_collection(text_backend.collection)
    bivec_qd_backend.bed_backend.qd_client.delete_collection(bed_backend.collection)

    # test HNSWBackend
    metadata_hnswb.load(text_embeddings, payloads=metadata_payloads)
    bivec_hnsw_backend = BiVectorBackend(metadata_hnswb, bed_hnswb)
    bivec_test(bivec_hnsw_backend, dist=True, rank=True)
    bivec_test(bivec_hnsw_backend, dist=True, rank=False)


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
    bed_collection,
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

    qd_search_backend = QdrantBackend(collection=bed_collection)
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

    # delete testing bed_collection
    db_interface.backend.qd_client.delete_collection(collection_name=bed_collection)

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
    bed_collection,
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

    qd_search_backend = QdrantBackend(collection=bed_collection)
    qd_search_backend.load(vectors=vecs, payloads=payloads)

    bed2vec = BED2Vec(r2v_hf_repo)

    # # construct a search interface
    db_interface = BED2BEDSearchInterface(qd_search_backend, bed2vec)
    db_search_result = db_interface.query_search(query_bed, 5, offset=0)
    for i in range(len(db_search_result)):
        assert isinstance(db_search_result[i], dict)

    # delete testing bed_collection
    db_interface.backend.qd_client.delete_collection(collection_name=bed_collection)

    # construct a search interface with file backend
    temp_data_dir = tmp_path_factory.mktemp("data")
    temp_idx_path = temp_data_dir / "testing_idx.bin"
    hnsw_backend = HNSWBackend(local_index_path=str(temp_idx_path))
    hnsw_backend.load(vectors=vecs, payloads=payloads)
    file_interface = Text2BEDSearchInterface(hnsw_backend, bed2vec)

    file_search_result = file_interface.query_search(query_bed, 5)
    for i in range(len(file_search_result)):
        assert isinstance(file_search_result[i], dict)
