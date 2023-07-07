#!/usr/bin/env python3
"""
bedfile embeding pipeline (test)
"""
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
from helpers import data_prepration_test, hash_bedfile
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
        Xs.append(X)
    return Xs


def label_preprocessing(path_word_embedding, label_prefix):
    labels = []
    label_vectors = []
    word_embedding = pd.read_csv(path_word_embedding, sep="\t", header=None)
    vectors = word_embedding[word_embedding[0].str.contains(label_prefix)]  # .reset_index()
    for l in range(len(vectors)):
        label_vectors.append((list(vectors.iloc[l])[1:]))
        labels.append(list(vectors.iloc[l])[0].replace(label_prefix, ""))
    return label_vectors, labels


def calculate_distance(X_files, X_labels, y_files, y_labels):
    X_files = np.array(X_files)
    X_labels = np.array(X_labels)
    distance_matrix = distance.cdist(X_files, X_labels, "cosine")
    df_distance_matrix = pd.DataFrame(distance_matrix)
    df_distance_matrix.columns = y_labels
    df_distance_matrix["file_label"] = y_files
    file_distance = pd.melt(
        df_distance_matrix,
        id_vars="file_label",
        var_name="search_term",
        value_name="score",
    )
    scaler = MinMaxScaler()
    file_distance["score"] = scaler.fit_transform(np.array(file_distance["score"]).reshape(-1, 1))
    return file_distance


def meta_preprocessing(meta):
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
    "-meta",
    "--meta_path",
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

    meta_df = pd.read_csv(args.meta_path)
    file_list = []
    for i in range(len(meta_df)):
        meta_data, assembly = meta_preprocessing(meta_df.iloc[i])
        file_list.append(meta_data)

    # define file path
    model = os.path.join(args.output_path, "starspace_model_{}".format(assembly))
    label_embed = os.path.join(args.output_path, "starspace_model_{}.tsv".format(assembly))
    docs = os.path.join(args.output_path, "documents_{}.txt".format(assembly))
    #     files = os.path.join(args.output_path, "filenames_{}.txt".format(assembly))
    doc_embed = os.path.join(args.output_path, "train_starspace_embed_{}.txt".format(assembly))
    dist = os.path.join(args.output_path, "similarity_score_{}.csv".format(assembly))

    embedding_labels, labels = label_preprocessing(label_embed, label_prefix)

    # Data prepration
    trained_documents = []
    with Pool(n_process) as p:
        trained_documents = p.starmap(data_prepration_test, [(x, universe) for x in file_list])
        p.close()
        p.join()
    print("Reading files done")

    df = pd.DataFrame(trained_documents, columns=["file_path", "context"])
    df = df.fillna(" ")
    df = df[df.context != " "]

    with open(docs, "w") as input_file:
        input_file.write("\n".join(df.context))
    input_file.close()

    print("Writing files done")

    starspace_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools",
        "StarSpace",
        "embed_doc",
    )

    output = check_output([starspace_path, model, docs]).decode("utf-8")

    with open(doc_embed, "w") as out:
        out.write(output)
        out.close()

    Xs = data_preprocessing(doc_embed)

    for i in range(len(file_list)):
        y = ",".join(file_list[i].split(",")[1:])
        X = Xs[i]

        df_similarity = calculate_distance(X, embedding_labels, y, labels)

        df_similarity["filename"] = [file_list[i].split(",")[0]] * len(labels)

        df_similarity = df_similarity[["filename", "file_label", "search_term", "score"]]

        # filter res by dist threshold
        thresh = 0.5
        df_similarity = df_similarity[df_similarity["score"] < thresh].reset_index(drop=True)

        if os.path.exists(dist):
            df_similarity.to_csv(dist, header=False, index=None, mode="a")
        else:
            df_similarity.to_csv(dist, index=False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Pipeline aborted.")
        sys.exit(1)
