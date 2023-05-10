#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import os
import glob
from multiprocessing import Pool
import pybedtools
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
import argparse
import datetime
from subprocess import check_output
from bbconf.const import *
from bbconf import BedBaseConf

def data_prepration_test(path_file_label, univ):
    path_file_label = path_file_label.split(",")
    path_file = path_file_label[0]
    if os.path.exists(path_file):
        try:
            df = pybedtools.BedTool(path_file)
            file_regions = univ.intersect(df, wa=True)
            file_regions.columns = ["chrom", "start", "end"]
            if len(file_regions) == 0:
                return " "
            file_regions = file_regions.to_dataframe().drop_duplicates()
            file_regions["region"] = (
                file_regions["chrom"]
                + "_"
                + file_regions["start"].astype(str)
                + "_"
                + file_regions["end"].astype(str)
            )
            return [path_file, " ".join(list(file_regions["region"]))]
        except Exception:
            print("Error in reading file: ", path_file)
            return [path_file, " "]
    else:
        return [path_file, " "]
    
def data_preprocessing(path_embeded_document):
    document_embedding = pd.read_csv(path_embeded_document, header=None)
    document_embedding = document_embedding[5:].dropna()
    document_embedding = document_embedding[::2].reset_index()
    document_embedding = document_embedding[0].str.split(" ", expand=True)
    X = document_embedding[list(document_embedding)[0:-1]].astype(float).values
    return X


def label_preprocessing(path_word_embedding, label_prefix):
    labels = []
    label_vectors = []
    word_embedding = pd.read_csv(path_word_embedding, sep="\t", header=None)
    vectors = word_embedding[
        word_embedding[0].str.contains(label_prefix)
    ] 
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
    df_distance_matrix["filename"] = y_files
    file_distance = pd.melt(
        df_distance_matrix,
        id_vars="filename",
        var_name="search_term",
        value_name="score",
    )
    scaler = MinMaxScaler()
    file_distance["score"] = scaler.fit_transform(
        np.array(file_distance["score"]).reshape(-1, 1)
    )
    file_distance["score"] = 1 - file_distance["score"]
    return file_distance

def top10(score_list):
    return (score_list[0:10])



bedbase_config = False

data_path='/project/shefflab/StarSpace_models/samples/'
path_starspace='../'
universe_path = '/project/shefflab/StarSpace_models/tiles1000.hg19.bed'
model_path ='/project/shefflab/StarSpace_models/'
output_path='./'
assembly='hg19'
label_prefix = "__label__"
n_process = 28



def return_simscores(file_list, label):
    universe = pybedtools.BedTool(universe_path)

    model = os.path.join(model_path, "starspace_model_{}_{}".format(assembly, label))
    label_embed = os.path.join(
        model_path, "starspace_model_{}_{}.tsv".format(assembly, label)
    )
    docs = os.path.join(output_path, "documents_{}.txt".format(assembly))
    doc_embed = os.path.join(
        output_path, "starspace_embed_{}.txt".format(assembly)
    )
    
    label_embeddings, labels = label_preprocessing(label_embed, label_prefix)

    # Data prepration
    documents = []
    with Pool(n_process) as p:
        documents = p.starmap(
            data_prepration_test, [(x, universe) for x in file_list]
        )
        p.close()
        p.join()

    df = pd.DataFrame(documents, columns=["file_path", "context"])
    
    df = df.fillna(" ")
    df = df[df.context != " "]

    with open(docs, "w") as input_file:
        input_file.write("\n".join(df.context))
    input_file.close()

    starspace_path = os.path.join(
        path_starspace,
        "tools",
        "Starspace",
        "embed_doc",
    )

    output = check_output([starspace_path, model, docs]).decode("utf-8")

    with open(doc_embed, "w") as out:
        out.write(output)
        out.close()

    file_embeddings = data_preprocessing(doc_embed)
    df_similarity = calculate_distance(file_embeddings, label_embeddings, file_list, labels)
    df_similarity = df_similarity[["filename", "search_term", "score"]]
    df_similarity['term_score'] = list(zip(df_similarity.search_term, df_similarity.score))
    
    os.remove(doc_embed)
    os.remove(docs)
    
    return df_similarity.sort_values(by = ['filename','score'], ascending = False).groupby(['filename'])['term_score'].apply(list).apply(top10).to_dict()



def run_bedembed(data_path):
    file_list = glob.glob(data_path+'*')
    similarity_antibody = return_simscores(file_list, 'antibody')
    print(similarity_antibody)
    similarity_cell_type = return_simscores(file_list, 'cell_type')
    print(similarity_cell_type)
    return similarity_antibody, similarity_cell_type
    

run_bedembed(data_path)
