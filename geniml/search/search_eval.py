import json
from typing import Dict, List, Union

from huggingface_hub import hf_hub_download

from .backends import HNSWBackend
from .const import HF_INDEX, HF_METADATA, HF_PAYLOADS
from .interfaces import Text2BEDSearchInterface
from .query2vec import Text2Vec


def anecdotal_search_from_hf_data(
    query: str, dataset_repo: str, search_model_repo: str, text_embed_model_repo: str, k: int = 10
) -> List[Dict[str, Union[float, int, Dict[str, str]]]]:
    """Test retrieval performance of a trained search model on a Hugging Face dataset.

    Args:
        query: user input search term
        dataset_repo: Hugging Face repository of the dataset
        search_model_repo: Hugging Face repository of the search model
        text_embed_model_repo: Hugging Face repository of the text encoder model
        k: number of results to return

    Returns:
        A list of dictionaries containing search scores (distance to the mapped query vector)
        and search result metadata.
    """

    # download embedding vector backends and metadata from huggingface
    index_path = hf_hub_download(dataset_repo, HF_INDEX, repo_type="dataset")
    payloads_path = hf_hub_download(dataset_repo, HF_PAYLOADS, repo_type="dataset")
    metadata_path = hf_hub_download(dataset_repo, HF_METADATA, repo_type="dataset")

    # evaluation backend
    eval_backend = HNSWBackend(local_index_path=index_path, payloads=payloads_path)

    # load metadata
    with open(metadata_path, "r") as f:
        metadata_dict = json.load(f)

    text2vec = Text2Vec(text_embed_model_repo, search_model_repo)
    search_interface = Text2BEDSearchInterface(eval_backend, text2vec)

    search_results = search_interface.query_search(query, k, with_payload=True, with_vectors=False)

    # curate output dictionary
    result_files_id_dict = {
        search_results[i]["payload"]["file"]: i for i in range(len(search_results))
    }
    for attribute in metadata_dict:
        for metadata in metadata_dict[attribute]:
            for file in metadata_dict[attribute][metadata]:
                try:
                    search_results[result_files_id_dict[file]]["payload"][attribute] = metadata
                except:
                    continue

    return search_results
