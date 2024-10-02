import json
import os
import tempfile
from typing import Dict

import numpy as np
from huggingface_hub import hf_hub_download

from ..backends import BiVectorBackend, HNSWBackend
from ..const import TEXT_ENCODER_REPO
from ..interfaces import BiVectorSearchInterface


def load_json(json_path: str) -> Dict:
    """
    Load metadata stored in json files

    :param json_path: path to json file
    :return: dictionary stored in the json file
    """
    with open(json_path, "r") as f:
        result = json.load(f)
    return result


def load_vectors(npz_path, vec_key="vectors") -> np.ndarray:
    """
    Load vectors stored in .npz file

    :param npz_path: path to the npz file
    :param vec_key: storage key of vector in the npz file
    :return: the stored vectors
    """
    data = np.load(npz_path)
    return data[vec_key]


def hf_bivec_search(query, repo: str = TEXT_ENCODER_REPO, limit=5, p=1.0, q=1.0, rank=True):
    """
    Demo using data loaded onto huggingface dataset

    :param query: free form query terms
    :param repo: the huggingface repository of text encoder model
    :param limit:see docstring of geniml.search.backend.BiVectorBackend
    :param p:
    :param q:
    :param rank:
    :return: the search result from demo dataset on huggingface
    """

    # download files from huggingface dataset
    bed_embeddings_path = hf_hub_download(repo, "bed_embeddings.npz", repo_type="dataset")
    file_id_path = hf_hub_download(repo, "file_id.json", repo_type="dataset")
    metadata_path = hf_hub_download(repo, "file_key_metadata.json", repo_type="dataset")
    metadata_match_path = hf_hub_download(repo, "metadata_id_match.json", repo_type="dataset")
    text_embeddings_path = hf_hub_download(repo, "text_embeddings.npz", repo_type="dataset")

    # load data from downloaded files
    file_id_dict = load_json(file_id_path)
    metadata_dict = load_json(metadata_path)
    metadata_match_dict = load_json(metadata_match_path)

    bed_data = np.load(bed_embeddings_path)
    bed_embeddings = bed_data["vectors"]
    bed_names = list(bed_data["names"])

    bed_name_idx = {value: index for index, value in enumerate(bed_names)}

    text_data = np.load(text_embeddings_path)

    text_embeddings = text_data["vectors"]
    text_annotations = list(text_data["texts"])

    bed_payloads = []
    bed_vecs = []

    # vectors and payloads for metadata backend
    for i in range(len(file_id_dict)):
        bed_embedding_id = bed_name_idx[file_id_dict[str(i)]]
        bed_vecs.append(bed_embeddings[bed_embedding_id])
        bed_payloads.append(
            {"name": file_id_dict[str(i)], "metadata": metadata_dict[file_id_dict[str(i)]]}
        )

    # payloads for bed file backend
    text_payloads = []
    for annotation in text_annotations:
        text_payloads.append(
            {"term": annotation, "matched_files": metadata_match_dict[annotation]}
        )

    # backends in temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # backend for BED file embedding vectors
        bed_backend = HNSWBackend(local_index_path=os.path.join(temp_dir, "bed.bin"), dim=100)
        bed_backend.load(vectors=np.array(bed_vecs), payloads=bed_payloads)

        # backend for metadata embedding vectors
        text_backend = HNSWBackend(local_index_path=os.path.join(temp_dir, "text.bin"), dim=384)
        text_backend.load(vectors=np.array(text_embeddings), payloads=text_payloads)

        # combined bi-vector search backend
        search_backend = BiVectorBackend(text_backend, bed_backend)

        # search interface
        search_interface = BiVectorSearchInterface(
            backend=search_backend, query2vec="sentence-transformers/all-MiniLM-L6-v2"
        )

        result = search_interface.query_search(
            query=query,
            limit=limit,
            with_payload=True,
            p=p,
            q=q,
            with_vectors=False,
            distance=True,  # HNSWBackend returns result by distance instead of similarity
            rank=rank,
        )

        return result
