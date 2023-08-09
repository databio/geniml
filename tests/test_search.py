import pytest
from gitk.search.search import QdrantBackend
from gitk.region2vec.main import Region2VecExModel
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import pprint


@pytest.fixture
def bed_folder():
    return "./data/hg38_sample"


def test_QdrantBackend(bed_folder):
    # r2v_model = Region2VecExModel(r2v_hf_repo)
    config = VectorParams(size=100, distance=Distance.COSINE)
    collection = "hg38_sample"
    file_names = os.listdir(bed_folder)
    embeddings = np.random.random((len(file_names), 100))

    qd_search_backend = QdrantBackend(config, collection)
    qd_search_backend.load(embeddings, file_names)
    search_result = qd_search_backend.search(np.random.random(100,), 5)
    assert isinstance(search_result, list)
    assert isinstance(search_result[0].id, int)
    assert isinstance(search_result[0].payload, dict)




