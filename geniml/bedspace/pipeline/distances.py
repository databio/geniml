import logging
import os
import datetime
import numpy as np
import pandas as pd
import pybedtools
from sklearn.preprocessing import MinMaxScaler

from helpers import meta_preprocessing, bed2vec, get_label_embedding, get_embedding_matrix, calculate_distance
from ..const import CACHE_DIR, DEFAULT_THRESHOLD, PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def main(
    type: str, 
    input: str,
    path_to_starsapce: str,
    metadata_test: str,
    metadata_train: str,
    universe: str,
    project_name: str,
    output: str,
    labels: str,
    files: str,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    Main function for the distance pipeline

    :param input: Path to the trained starSpace model
    :param path_to_starsapce: Path to starspace folder (must be prebuilt)
    :param metadata_test: Path to test metadata
    :param metadata_train: Path to train metadata
    :param universe: Path to universe file
    :param project_name: Name of the project
    :param output: Path to output folder to save the distance files
    :param labels: Labels string (cell_type,target)
    :param files: Path to files to be embedded (the data)
    :param threshold: Threshold for filtering
    """
    _LOGGER.info("Running distances...")
    _LOGGER.info("Start", datetime.datetime.now())

    # PLACE CODE FOR RUNNING DISTANCES HERE
    label_prefix = "__label__"
    temp_path = CACHE_DIR

    ## load input files
    universe = pybedtools.BedTool(universe)
    file_list_train = meta_preprocessing(pd.read_csv(metadata_train), labels, files)
    file_list_test = meta_preprocessing(pd.read_csv(metadata_test), labels, files)

    ## calculate region set (sample) to label distance
    #  define file path 
    label_embed = os.path.join("{}.tsv".format(input))
    docs = os.path.join(temp_path, "test_documents.txt")
    doc_embed_query = os.path.join(output, "{}_starspace_embed.txt".format(project_name))
    distance_file_path_rl = os.path.join(output, "raw_cosdist_rl.csv")
    similarity_file_path_rl = os.path.join(output, "similarity_score_rl.csv")

    # get trained label embedding
    embedding_labels, labels_l = get_label_embedding(label_embed, label_prefix)
    # predict sample embedding 
    bed2vec(
        file_list_test, universe, input, docs,doc_embed_query, path_to_starsapce
    )
    # create sample&label embedding matrix
    query_vectors = get_embedding_matrix(doc_embed_query)
    # calculate cosine distance
    df_similarity = calculate_distance(query_vectors, embedding_labels, file_list_test, labels_l, "rl")
    df_similarity["filename"] = [file_list_test[i].split(",")[0] for i in range(len(file_list_test))] * len(
        labels_l
    )
    df_similarity = df_similarity[["filename", "file_label", "search_term", "score"]]
    # saving the raw distance table
    df_similarity.to_csv(distance_file_path_rl, index=False)
    # scale the distance from 0 to 1
    scaler = MinMaxScaler()
    df_similarity["score"] = scaler.fit_transform(np.array(df_similarity["score"]).reshape(-1, 1))
    df_similarity = df_similarity[df_similarity["score"] > threshold].reset_index(drop=True)
    # convert distance to similarity
    df_similarity["score"] = 1 - df_similarity["score"]
    # saving the scaled filtered similarity table
    df_similarity.to_csv(similarity_file_path_rl, index=False)


    ## calculate region set (test sample) to region set distance (train sample)
    #  define file path
    distance_file_path_rr = os.path.join(output, "similarity_score_rr.csv")
    docs_db = os.path.join(temp_path, "train_documents.txt")
    doc_embed_db = os.path.join(output, "{}_train_starspace_embed.txt".format(project_name))
    # predict training sample embedding
    bed2vec(
        file_list_train, universe, input, docs_db,doc_embed_db, path_to_starsapce
    )
    # create sample&label embedding matrix
    db_vectors = get_embedding_matrix(doc_embed_db)
    # calculate cosine distance
    df_similarity = calculate_distance(db_vectors, query_vectors, file_list_train, file_list_test, "rr")
    df_similarity = df_similarity[["db_file", "test_file", "score"]]
    # convert distance to similarity
    df_similarity["score"] = 1 - df_similarity["score"]
    # saving the scaled similarity table
    df_similarity.to_csv(distance_file_path_rr, index=False)

    _LOGGER.info("Test sample distances prediction done.")
    _LOGGER.info("End", datetime.datetime.now())

