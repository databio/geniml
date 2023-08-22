import os
from typing import List, Tuple, Dict
import random
import numpy as np
import tensorflow as tf
from .const import *
from dataclasses import dataclass, replace
from ..io import RegionSet
from ..region2vec import Region2VecExModel
from sentence_transformers import SentenceTransformer


@dataclass
class RegionsetInfo:
    """
    Store the information of a bed file, its metadata, and embeddings
    """
    file_name: str  # the name of the bed file
    metadata: str  # the metadata of the bed file
    region_set: RegionSet  # the RegionSet that contains intervals in that bed file, not tokenized
    metadata_embedding: np.ndarray  # the embedding vector of the metadata by sentence transformer
    region_set_embedding: np.ndarray  # the embedding vector of region set


def build_RegionsetInfo_list(bed_folder: str,
                             metadata_path: str,
                             r2v_model: Region2VecExModel,
                             st_model: SentenceTransformer) -> List[RegionsetInfo]:
    """
    With each bed file in the given folder and its matching metadata from the meadata file,
    create a RegionsetInfo with each, and return the list containing all.

    :param bed_folder:
    :param metadata_path:
    :param r2v_model:
    :param st_model:
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
    i = 0
    j = 0

    while i < len(metadata_lines):

        # read the line of metadata
        metadata_line = metadata_lines[i]
        # get the name of the interval set
        set_name = metadata_line.split("\t")[0]

        if j < len(file_name_list) and file_name_list[j].startswith(set_name):
            bed_file_name = file_name_list[j]
            bed_file_path = os.path.join(bed_folder, bed_file_name)
            bed_metadata = metadata_line_process(metadata_line)
            region_set = RegionSet(bed_file_path)
            metadata_embedding = st_model.encode(bed_metadata)
            region_set_embedding = np.nanmean(r2v_model.encode(region_set), axis=0)
            bed_metadata_dc = RegionsetInfo(bed_file_name,
                                            bed_metadata,
                                            region_set,
                                            metadata_embedding,
                                            region_set_embedding)
            output_list.append(bed_metadata_dc)
            j += 1

        # end the loop if all
        if j == len(file_name_list):
            break

        i += 1

    # print a message if not all bed files are matched to metadata rows
    if i < j:
        print("An incomplete list will be returned, some files cannot be matched to any rows by first column")

    return output_list


def RegionsetInfo_list_to_vectors(ri_list: List[RegionsetInfo]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    Y = []
    for ri in ri_list:
        # X: metadata embedding
        X.append(ri.metadata_embedding)
        # Y: bed file embedding
        Y.append(ri.region_set_embedding)
    return np.array(X), np.array(Y)


def search_backend_upload(ri_list: List[RegionsetInfo]) -> Tuple[np.ndarray, Dict[str, str]]:
    embeddings = []
    labels = []
    for ri in ri_list:

        # X: metadata embedding
        embeddings.append(ri.region_set_embedding)
        # Y: bed file embedding
        labels.append({"name": ri.file_name,
                       "metadata": ri.metadata})

    return np.array(embeddings), labels


def data_split(full_list: List,
               train_p: float = DEFAULT_TRAIN_P,
               valid_p: float = DEFAULT_VALIDATE_P,
               seed_index: int = 10) -> Tuple[List, List, List]:
    """
    With a given folder of data, this function split the files
    into training set, validating set, and testing set.

    :param full_list: list of data
    :param train_p: proportion of files for training set
    :param valid_p: proportion of files for validating set
    and files will be copied to them.
    :param seed_index: index for random seed
    :return: lists of file names for training set, validating set, and training set.
    """

    # split training data and testing data
    train_size = int(len(full_list) * train_p)
    validate_size = int(len(full_list) * valid_p)
    random.seed(seed_index)
    train_list = random.sample(full_list, train_size)
    validate_list = random.sample([content for content in full_list if content not in train_list], validate_size)
    test_list = [content for content in full_list if (content not in train_list and content not in validate_list)]

    return train_list, validate_list, test_list


def update_BedMetadata_list(old_list: List[RegionsetInfo],
                            r2v_model: Region2VecExModel) -> List[RegionsetInfo]:
    """
    With an old list of RegionsetInfo, re-embed the region set with a new region2vec model,
    then return the list of new RegionsetInfo with re-embedded region set vectors.
    :param old_list:
    :param r2v_model:
    :return:
    """
    new_list = []
    for dc in old_list:
        new_dc = replace(dc, region_set_embedding=r2v_model.encode(dc.region_set))
        new_list.append(new_dc)

    return new_list


def metadata_line_process(metadata_line: str) -> str:
    """
    remove formatting characters and interval set name from metadata

    :param set_name: name of interval set
    :param metadata_line: the metadata text
    :return: the metadata text without interval set name and formatting characters
    """

    metadata_line = metadata_line.replace("\t", " ")
    metadata_line = metadata_line.replace("\n", "")

    return metadata_line


"""
ls | shuf -n 20 | xargs -I '{}' cp '{}' /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_sample

for file in /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_sample/*.bed; do
    key=$(basename "$file" | cut -d. -f1)
    grep -w "$key" experimentList_sorted_hg38.tab >> /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_metadata_sample.tab
done

shuf -n 20 experimentList_sorted_hg38.tab >> /home/claudehu/Desktop/repo/geniml_dev/tests/data/hg38_metadata_sample.tab

sort -k1,1 -t$'\t' hg38_metadata_sample.tab > hg38_metadata_sample_sorted.tab


"""
