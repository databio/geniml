import logging
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Union

import numpy as np
import peppy
import torch
from fastembed.embedding import FlagEmbedding
from torch.utils.data import DataLoader, TensorDataset

from ..io import RegionSet
from ..region2vec import Region2VecExModel
from ..search.backends import HNSWBackend, QdrantBackend
from .const import *

_LOGGER = logging.getLogger(MODULE_NAME)


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


def build_regionset_info_list_from_PEP(
    yaml_path: str,
    col_names: List[str],
    r2v_model: Region2VecExModel,
    nl_embed: FlagEmbedding,
    with_regions: bool = False,
    bed_vec_necessary: bool = True,
) -> List[RegionSetInfo]:
    """
    With each bed file and its metadata and from a Portable Encapsulated Projects (PEP),
    create a RegionSetInfo with each, and return the list containing all.

    :param yaml_path: the path to the yaml file, which is the metadata validation framework of the PEP
    :param col_names: the name of needed columns in the metadata csv
    :param r2v_model: a Region2VecExModel that can embed region sets
    :param nl_embed: a FlagEmbedding that can embed metadata
    :param with_regions: if false, no RegionSetInfo in the output list will contain the RegionSet object from the bedfile (replaced by None).
    :param bed_vec_necessary: whether the embedding vector of a bed file has to be valid (not None)
    to be included into the list

    :return: a list of RegionSetInfo
    """

    output_list = []
    project = peppy.Project(yaml_path)
    # assure the column list is not empty
    if len(col_names) == 0:
        _LOGGER.error(
            "ValueError: please give the name of at least one column in the metadata csv"
        )
    for sample in project.samples:
        # get the path to the bed file
        bed_file_path = sample.output_file_path
        if not os.path.exists(bed_file_path):
            _LOGGER.warning(f"Warning: {bed_file_path}' does not exist")
            continue
        bed_file_name = os.path.split(bed_file_path)[1]
        # get the metadata
        bed_metadata = ";".join(sample[col] for col in col_names if sample[col] is not None)
        region_set = RegionSet(bed_file_path)
        metadata_embedding = next(nl_embed.embed(bed_metadata))
        region_set_embedding = r2v_model.encode(region_set, pool="mean", return_none=False)
        if region_set_embedding is None and bed_vec_necessary:
            _LOGGER.warning(f"Warning: {bed_file_name}'s embedding is None, exclude from dataset")
            continue
        if not with_regions:
            region_set = None
        bed_metadata_dc = RegionSetInfo(
            bed_file_name, bed_metadata, region_set, metadata_embedding, region_set_embedding
        )
        output_list.append(bed_metadata_dc)
    return output_list


def build_regionset_info_list_from_files(
    bed_folder: str,
    metadata_path: str,
    r2v_model: Region2VecExModel,
    nl_embed: FlagEmbedding,
    with_regions: bool = False,
    bed_vec_necessary: bool = True,
) -> List[RegionSetInfo]:
    """
    With each bed file in the given folder and its matching metadata from the metadata file,
    create a RegionSetInfo with each, and return the list containing all.

    :param bed_folder: folder where bed files are stored
    :param metadata_path: path to the metadata file
    :param r2v_model: a Region2VecExModel that can embed region sets
    :param nl_embed: a model that can embed natural language
    :param with_regions: if false, no RegionSetInfo in the output list will contain the RegionSet object from the bedfile (replaced by None).
    :param bed_vec_necessary: whether the embedding vector of a bed file has to be valid (not None)
    to be included into the list

    :return: a list of RegionSetInfo
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
        # end loop when all bed files have been processed
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
            metadata_embedding = next(nl_embed.embed(bed_metadata))
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
        _LOGGER.warning(
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
            _LOGGER.warning(
                f"WARNING: {ri.file_name}'s embedding has shape of {ri.region_set_embedding.shape}, exclude from dataset"
            )
            # _LOGGER.error()
            continue
        X.append(ri.metadata_embedding)
        # Y: bed file embedding
        Y.append(ri.region_set_embedding)
    return np.array(X), np.array(Y)


def prepare_vectors_for_database(
    ri_list: List[RegionSetInfo],
    filename_key: str = DEFAULT_FILENAME_KEY,
    metadata_key: str = DEFAULT_METADATA_KEY,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    With a given list of RegionSetInfo, returns one np.ndarray representing bed files embeddings,
    and one list of dictionary that stores names of bed files and metadata,
    used as data preprocessing for upload to search backend (geniml.search)

    :param ri_list: RegionSetInfo list

    :return: one np.ndarray with shape (n, <Region2vec embedding dimension>),
    and one list of dictionary in the format of:
    {
        <filename_key>: <bed file name>,
        <metadata_key>>: <region set metadata>
    }
    """
    embeddings = []
    labels = []
    for ri in ri_list:
        embeddings.append(ri.region_set_embedding)
        # file name and metadata
        labels.append({filename_key: ri.file_name, metadata_key: ri.metadata})

    return np.array(embeddings), labels


def vectors_from_backend(
    search_backend: Union[HNSWBackend, QdrantBackend],
    encoding_model: FlagEmbedding,
    payload_key: str = DEFAULT_PAYLOAD_KEY,
    vec_key: str = DEFAULT_VECTOR_KEY,
    metadata_key: str = DEFAULT_METADATA_KEY,
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
    :param encoding_model: model that can encode natural language, like FastEmbedding

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
        metadata_vec = next(encoding_model.embed(metadata))
        X.append(metadata_vec)

    # return output in np.ndarray format
    return np.array(X), np.array(Y)


def arrays_to_torch_dataloader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = DEFAULT_DATALOADER_SHUFFLE,
) -> DataLoader:
    """
    Based on https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

    Store np.ndarray of X and Y into a torch.DataLoader
    """
    tensor_X = torch.from_numpy(dtype_check(X))
    tensor_Y = torch.from_numpy(dtype_check(Y))
    my_dataset = TensorDataset(tensor_X, tensor_Y)  # create your datset

    return DataLoader(my_dataset, batch_size=batch_size, shuffle=shuffle)


def dtype_check(vecs: np.ndarray) -> np.ndarray:
    """
    Since the default float in np is float64, but in pytorch tensor it's float32,
    to avoid errors, the dtype will be switched

    :return: np.ndarray with dtype of float32
    """
    if not isinstance(vecs.dtype, type(np.dtype("float32"))):
        vecs = vecs.astype(np.float32)

    return vecs
