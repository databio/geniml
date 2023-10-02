import logging
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer, models

from ..const import PKG_NAME
from ..io import RegionSet
from ..region2vec import Region2VecExModel
from ..search.backends import HNSWBackend, QdrantBackend
from .const import *

_LOGGER = logging.getLogger(PKG_NAME)


@dataclass
class RegionSetInfo:
    """
    Store the information of a bed file, its metadata, and embeddings
    """

    file_name: str  # the name of the bed file
    metadata: str  # the metadata of the bed file
    region_set: Union[
        RegionSet, None
    ]  # the RegionSet that contains intervals in that bed file, not tokenized
    metadata_embedding: np.ndarray  # the embedding vector of the metadata by sentence transformer
    region_set_embedding: np.ndarray  # the embedding vector of region set


def build_regionset_info_list(
    bed_folder: str,
    metadata_path: str,
    r2v_model: Region2VecExModel,
    st_model: SentenceTransformer,
    with_regions: bool = False,
    bed_vec_necessary: bool = True,
) -> List[RegionSetInfo]:
    """
    With each bed file in the given folder and its matching metadata from the metadata file,

    create a RegionSetInfo with each, and return the list containing all.

    :param bed_folder: folder where bed files are stored
    :param metadata_path: path to the metadata file
    :param r2v_model: a Region2VecExModel that can embed region sets
    :param st_model: a SentenceTransformer model that can embed metadata
    :param with_regions: whether the RegionSet will be added into each dataclass
    :param bed_vec_necessary: whether the embedding vector of a bed file has to be valid (not None)
    to be included into the list
    :return:
    """

    file_name_list = os.listdir(bed_folder)
    file_name_list.sort()

    output_list = []

    # read the lines from the metadata file
    with open(metadata_path) as m:
        metadata_lines = m.readlines()

    # index to traverse metadata and file list
    # make sure metadata is sorted by the name of interval set
    # this can be done by this command
    # sort -k1 1 metadata_file >  new_metadata_file
    metadata_file_index = 0
    bed_folder_index = 0

    while metadata_file_index < len(metadata_lines):
        # end the loop if all bed files has been go over
        if bed_folder_index == len(file_name_list):
            break
        # read the line of metadata
        metadata_line = metadata_lines[metadata_file_index]
        # get the name of the interval set
        set_name = metadata_line.split("\t")[0]

        if bed_folder_index < len(file_name_list) and file_name_list[bed_folder_index].startswith(
            set_name
        ):
            bed_file_name = file_name_list[bed_folder_index]
            bed_file_path = os.path.join(bed_folder, bed_file_name)
            bed_metadata = clean_escape_characters(metadata_line)
            region_set = RegionSet(bed_file_path)
            metadata_embedding = st_model.encode(bed_metadata)
            region_set_embedding = r2v_model.encode(region_set, pool="mean", return_none=False)
            if region_set_embedding is None and bed_vec_necessary:
                _LOGGER.info(f"{bed_file_name}'s embedding is None, exclude from dataset")
                bed_folder_index += 1
                metadata_file_index += 1
                continue
            if not with_regions:
                region_set = None
            bed_metadata_dc = RegionSetInfo(
                bed_file_name, bed_metadata, region_set, metadata_embedding, region_set_embedding
            )
            output_list.append(bed_metadata_dc)
            bed_folder_index += 1

        metadata_file_index += 1
        # print a message if not all bed files are matched to metadata rows
    if metadata_file_index < bed_folder_index:
        _LOGGER.info(
            "An incomplete list will be returned, some files cannot be matched to any rows by first column"
        )

    return output_list


def update_bed_metadata_list(
    old_list: List[RegionSetInfo], r2v_model: Region2VecExModel
) -> List[RegionSetInfo]:
    """
    With an old list of RegionSetInfo, re-embed the region set with a new Region2Vec model,
    then return the list of new RegionSetInfo with re-embedded region set vectors.
    :param old_list:
    :param r2v_model:
    :return:
    """
    new_list = []
    for region_set_info in old_list:
        # update reach RegionSetInfo with new embedding
        new_ri = replace(
            region_set_info, region_set_embedding=r2v_model.encode(region_set_info.region_set)
        )
        new_list.append(new_ri)

    return new_list


def clean_escape_characters(metadata_line: str) -> str:
    """
    Remove formatting characters from metadata

    :param metadata_line: the metadata text
    :return: the metadata text without interval set name and formatting characters
    """

    metadata_line = metadata_line.replace("\t", " ")
    metadata_line = metadata_line.replace("\n", "")

    return metadata_line


def region_info_list_to_vectors(ri_list: List[RegionSetInfo]) -> Tuple[np.ndarray, np.ndarray]:
    """
    With a given list of RegionSetInfo, returns two np.ndarrays,
    one represents embeddings of bed files, the other represents embedding of metadata,
    used as data preprocessing for fitting models.

    :param ri_list: RegionSetInfo list
    :return: two np.ndarray with shape (n, <embedding dimension>)
    """
    X = []
    Y = []
    for ri in ri_list:
        # X: metadata embedding
        if ri.region_set_embedding is None:
            _LOGGER.info(f"{ri.file_name}'s embedding is None, exclude from dataset")
            continue
        if ri.region_set_embedding.shape != DEFAULT_EMBEDDING_DIM:
            _LOGGER.info(
                f"{ri.file_name}'s embedding has shape of {ri.region_set_embedding.shape}, exclude from dataset"
            )
            continue
        X.append(ri.metadata_embedding)
        # Y: bed file embedding
        Y.append(ri.region_set_embedding)
    return np.array(X), np.array(Y)


def prepare_vectors_for_database(
    ri_list: List[RegionSetInfo],
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    With a given list of RegionSetInfo, returns one np.ndarray representing bed files embeddings,
    and one list of dictionary that stores names of bed files and metadata,
    used as data preprocessing for upload to search backend (geniml.search)

    :param ri_list: RegionSetInfo list
    :return: one np.ndarray with shape (n, <Region2vec embedding dimension>),
    and one list of dictionary in the format of:
    {
        "name": <bed file name>,
        "metadata": <region set metadata>
    }
    """
    embeddings = []
    labels = []
    for ri in ri_list:
        # region set embedding
        embeddings.append(ri.region_set_embedding)
        # file name and metadata
        labels.append({"name": ri.file_name, "metadata": ri.metadata})

    return np.array(embeddings), labels


def vectors_from_backend(
    search_backend: Union[HNSWBackend, QdrantBackend],
    encoding_model: SentenceTransformer,
    payload_key: str = "payload",
    vec_key: str = "vector",
    metadata_key: str = "metadata",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load vectors from a search backend's qdrant client or hnsw index
    and the metadata from payloads, then encode metadata into vectors.

    Return the two np.ndarrays, one contains vectors from search backend,
    and one contains matching metadata embedding vectors.

    :param search_backend: the search backend where vectors are stored
    :param payload_key: the key of payload in the retrieval output
    :param vec_key: the key of vector in the retrieval output
    :param metadata_key: the key of metadata content in the payload dictionary
    :param encoding_model: model that can encode natural language, like SentenceTransformer
    :return: two np.ndarray with shape (n, <embedding dimension>)
    """
    X = []
    Y = []
    n = len(search_backend)
    for i in range(n):
        # retrieve vector and metadata/payload
        vec_info = search_backend.retrieve_info(i, with_vec=True)
        # add vector to output list
        vec = vec_info[vec_key]
        Y.append(vec)
        # embed metadata and add embedding vector to output list
        metadata = vec_info[payload_key][metadata_key]
        metadata_vec = encoding_model.encode(metadata)
        X.append(metadata_vec)

    # return output in np.ndarray format
    return np.array(X), np.array(Y)


def bioGPT_sentence_transformer(model_name: str = DEFAULT_BIOGPT_MODEL) -> SentenceTransformer:
    """
    returns a sentencetransformer with specific model for biomedical text

    :param model_name: name of the biomedical text transformer model
    """
    word_embedding_model = models.Transformer(model_name)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
    )
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])
