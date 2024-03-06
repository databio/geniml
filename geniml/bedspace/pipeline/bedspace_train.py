"""
bedfile embeding pipeline (train)
"""

import argparse
import itertools
import os
import subprocess
import sys
from collections import Counter
from multiprocessing import Pool
from posixpath import join
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
import umap
from helpers import data_prepration
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()

parser.add_argument(
    "-star",
    "--starspace_path",
    default=None,
    type=str,
    required=True,
    help="Path to the StarSpace.",
)
parser.add_argument(
    "-i",
    "--input_path",
    default=None,
    type=str,
    required=True,
    help="Path to the bed files.",
)
parser.add_argument(
    "-g",
    "--genome",
    default=None,
    type=str,
    required=True,
    help="Genome assembly.",
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
    "-l",
    "--labels",
    default=None,
    type=str,
    required=True,
    help="columns use as label",
)
parser.add_argument(
    "-nof",
    "--no_files",
    default=None,
    type=int,
    required=True,
    help="Number of files to read.",
)
parser.add_argument(
    "-startline",
    "--start_line",
    default=None,
    type=int,
    required=True,
    help="The start line of metadata.",
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
    "-dim",
    "--dim",
    default=None,
    type=str,
    required=True,
    help="Embedding dimension.",
)
parser.add_argument(
    "-epochs",
    "--no_epochs",
    default=None,
    type=str,
    required=True,
    help="The start line of metadata.",
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    default=None,
    type=str,
    required=True,
    help="Path to output directory to store file.",
)

args = parser.parse_args()
d = vars(parser.parse_args())

if "labels" in d.keys():
    args.labels = [str(s.strip()) for s in d["labels"].split(",")]

n_process = 20
label_prefix = "__label__"
# StarSpace Parameters
universe = pybedtools.BedTool(args.univ_path)
# define file path
train_files = os.path.join(args.output_path, "documents_file_{}.txt".format(args.genome))
model = os.path.join(args.output_path, "starspace_model_{}".format(args.genome))


def meta_preprocessing(meta):
    cols = ["file_name"]
    cols.extend(args.labels)
    meta = meta[cols]
    meta["file_name"] = args.input_path + meta["file_name"]
    meta = meta.fillna("")
    meta[0] = meta.apply(",".join, axis=1)
    return meta[0]


def main():
    meta_data = meta_preprocessing(pd.read_csv(args.meta_path))
    file_list = list(meta_data)
    trained_documents = []
    with Pool(n_process) as p:
        trained_documents = p.starmap(
            data_prepration,
            [(x, universe) for x in file_list[args.start_line : args.start_line + args.no_files]],
        )
        p.close()
        p.join()

    print("Reading files done")

    df = pd.DataFrame(trained_documents, columns=["file_path", "context"])

    df = df.fillna(" ")
    df = df[df.context != " "]

    with open(train_files, "w") as input_file:
        input_file.write("\n".join(df.context))
    input_file.close()

    if os.path.exists(model):
        subprocess.Popen(
            [
                args.starspace_path,
                "train",
                "-trainFile",
                train_files,
                "-model",
                model,
                "-initModel",
                model,
                "-trainMode",
                "0",
                "-dim",
                args.dim,
                "-epoch",
                args.no_epochs,
                "-negSearchLimit",
                "5",
                "-thread",
                "20",
                "-lr",
                args.learning_rate,
            ]
        )
    else:
        subprocess.Popen(
            [
                args.starspace_path,
                "train",
                "-trainFile",
                train_files,
                "-model",
                model,
                "-trainMode",
                "0",
                "-dim",
                args.dim,
                "-epoch",
                args.no_epochs,
                "-negSearchLimit",
                "5",
                "-thread",
                "20",
                "-lr",
                args.learning_rate,
            ]
        )

    print("Train bedembed done.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Pipeline aborted.")
        sys.exit(1)
