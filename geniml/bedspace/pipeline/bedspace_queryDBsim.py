import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from random import shuffle
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
import umap
import yaml
from bbconf import BedBaseConf
from bbconf.const import *
from helpers import bed2vec, data_prepration_test, hash_bedfile
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ubiquerg import VersionInHelpParser


def data_preprocessing(path_embeded_document):
    document_embedding = pd.read_csv(path_embeded_document, header=None)
    document_embedding = document_embedding[0].str.split("__label__", expand=True)
    document_embedding[list(document_embedding)[1:]] = document_embedding[
        list(document_embedding)[1:]
    ].shift(1)
    document_embedding = document_embedding[5:].dropna()
    document_embedding = document_embedding[::2].reset_index()
    document_embedding = document_embedding[0].str.split(" ", expand=True)

    Xs = []
    for i in range(len(document_embedding)):
        X = document_embedding[i : i + 1]
        X = X[list(X)[0:-1]].astype(float)
        X = list(X.values)

        Xs.append(X[0])
    return np.array(Xs)


def calculate_distance(X_files, X_labels, y_files, y_labels):
    X_files = np.array(X_files)
    X_labels = np.array(X_labels)
    distance_matrix = distance.cdist(X_files, X_labels, "cosine")
    df_distance_matrix = pd.DataFrame(distance_matrix)
    df_distance_matrix.columns = y_labels
    df_distance_matrix["db_file"] = y_files
    file_distance = pd.melt(
        df_distance_matrix,
        id_vars="db_file",
        var_name="test_file",
        value_name="score",
    )
    scaler = MinMaxScaler()
    file_distance["score"] = scaler.fit_transform(np.array(file_distance["score"]).reshape(-1, 1))
    return file_distance


def meta_preprocessing(meta):
    #     print(meta)
    assembly = meta["genome"]
    labels = []
    for l in args.labels.split(","):
        if l in meta.index:
            labels.append(l)
    labels.insert(0, "file_name")
    meta = meta[labels]
    meta = meta.fillna("")
    meta["file_name"] = args.data_path + meta["file_name"]

    return ",".join(list(meta)), assembly


def meta(meta_path):
    meta_df = pd.read_csv(meta_path)
    file_list = []
    for i in range(len(meta_df)):
        meta_data, assembly = meta_preprocessing(meta_df.iloc[i])
        file_list.append(meta_data)

    return meta_data, assembly, file_list


parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_path",
    "--data_path",
    default=None,
    type=str,
    required=True,
    help="Path to the bed files.",
)
parser.add_argument(
    "-db_path",
    "--meta_path_db",
    default=None,
    type=str,
    required=True,
    help="Path to the metadata file.",
)
parser.add_argument(
    "-query_path",
    "--meta_path_query",
    default=None,
    type=str,
    required=True,
    help="Path to the metadata file.",
)
parser.add_argument(
    "-univ",
    "--univ_path",
    default=None,
    type=str,
    required=True,
    help="Path to the universe file.",
)
parser.add_argument(
    "-o",
    "--output_path",
    default="./",
    type=str,
    required=True,
    help="Path to output directory to store file.",
)
parser.add_argument(
    "--bedbase-config",
    dest="bedbase_config",
    type=str,
    default=None,
    help="a path to the bedbase configuratiion file",
)
parser.add_argument(
    "-l",
    "--labels",
    default=None,
    type=str,
    required=True,
    help="columns use as label",
)

args = parser.parse_args()

n_process = 20
label_prefix = "__label__"


def main():
    print("Start", datetime.datetime.now())
    universe = pybedtools.BedTool(args.univ_path)

    meta_data_db, assembly, file_list_db = meta(args.meta_path_db)
    meta_data_query, assembly, file_list_query = meta(args.meta_path_query)

    # define file path
    model = os.path.join(args.output_path, "starspace_model_{}".format(assembly))

    dist = os.path.join(args.output_path, "query_db_similarity_score_{}.csv".format(assembly))

    doc_embed_dB = bed2vec(file_list_db, universe, model, assembly, "DB", args.output_path)

    db_vectors = data_preprocessing(doc_embed_dB)

    print(db_vectors.shape)
    doc_embed_query = bed2vec(
        file_list_query, universe, model, assembly, "Query", args.output_path
    )

    query_vectors = data_preprocessing(doc_embed_query)

    print(query_vectors.shape)

    for i in range(len(query_vectors)):
        query_vector = query_vectors[i]

        df_similarity = calculate_distance(
            db_vectors, query_vectors, file_list_db, file_list_query
        )

        df_similarity = df_similarity[["db_file", "test_file", "score"]]

        # report to db
        if args.bedbase_config:
            print("Report distances to the database.")
            bbc = BedBaseConf(config_path=args.bedbase_config, database_only=True)
            for index, row in df_similarity.iterrows():
                bbc.report_distance(
                    bed_md5sum=md5sum,
                    bed_label=row["db_file"],
                    search_term=row["test_file"],
                    score=row["score"],
                )
        else:
            if os.path.exists(dist):
                df_similarity.to_csv(dist, header=False, index=None, mode="a")
            else:
                df_similarity.to_csv(dist, index=False)


#         print("End", datetime.datetime.now())


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Pipeline aborted.")
        sys.exit(1)
