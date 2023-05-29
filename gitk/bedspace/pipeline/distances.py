# import logging
import pybedtools
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from subprocess import check_output
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

from ..const import DEFAULT_THRESHOLD, PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def data_prepration_test(
    path_file_label, 
    univ
):
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
    
    
def bed2vec(file_list, universe, model, source, output_path, path_to_starsapce):

    docs = os.path.join(output_path, "documents.txt")

    doc_embed = os.path.join(
        output_path, "{}_starspace_embed.txt".format(source)
    )

    documents = []
    with Pool(16) as p:
        documents = p.starmap(data_prepration_test, [(x, universe) for x in file_list])
        p.close()
        p.join()
#     print("Reading files done")

    df = pd.DataFrame(documents, columns=["file_path", "context"])
    df = df.fillna(" ")
    df = df[df.context != " "]


    with open(docs, "w") as input_file:
        input_file.write("\n".join(df.context))
    input_file.close()


    starspace_path = os.path.join(
        path_to_starsapce,
        "embed_doc",
    )

    output = check_output([starspace_path, model, docs]).decode("utf-8")

    with open(doc_embed, "w") as out:
        out.write(output)
        out.close()
    return doc_embed

    
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
    return Xs


def label_preprocessing(path_word_embedding, label_prefix):
    labels = []
    label_vectors = []
    word_embedding = pd.read_csv(path_word_embedding, sep="\t", header=None)
    vectors = word_embedding[
        word_embedding[0].str.contains(label_prefix)
    ]  # .reset_index()
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
    df_distance_matrix["file_label"] = [y_files[i].split(",")[1] for i in range(len(y_files))] 
    file_distance = pd.melt(
        df_distance_matrix,
        id_vars="file_label",
        var_name="search_term",
        value_name="score",
    )
    scaler = MinMaxScaler()
    file_distance["score"] = scaler.fit_transform(
        np.array(file_distance["score"]).reshape(-1, 1)
    )
    return file_distance

def calculate_distance_qc(X_files, X_labels, y_files, y_labels):
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
    file_distance["score"] = scaler.fit_transform(
        np.array(file_distance["score"]).reshape(-1, 1)
    )
    return file_distance



def meta_preprocessing(meta, labels, data_path):
    cols = ["file_name"]
    cols.extend(labels.split(','))
    meta = meta[cols]
    meta.loc[:, "file_name"] = data_path + meta["file_name"]
    meta = meta.fillna("")
    meta[0] = meta.apply(",".join, axis=1)
    return meta[0]

def main(
    input: str,
    path_to_starsapce: str,
    metadata_test: str,
    metadata_train: str,
    universe: str,
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
    :param output: Path to output folder to save the distance files
    :param labels: Labels string (cell_type,target)
    :param files: Path to files to be embedded (the data)
    :param threshold: Threshold for filtering
    """
    _LOGGER.info("Running distances...")
    
    # PLACE CODE FOR RUNNING DISTANCES HERE
    n_process = 20
    label_prefix = "__label__"
    temp_path = './'
    
    
    universe = pybedtools.BedTool(universe)

    file_list = meta_preprocessing(pd.read_csv(metadata_test), labels, files)
    label_embed = os.path.join("{}.tsv".format(input))
    docs_test = os.path.join(temp_path, "documents_test.txt")
    doc_embed_test = os.path.join(temp_path, "testfiles_embed.txt")    
    distance_file_path_rl = os.path.join(output, "similarity_score_rl.csv")



    embedding_labels, labels_l = label_preprocessing(label_embed, label_prefix)
    
    doc_embed_query = bed2vec(file_list, universe, input, "testfiles", temp_path, path_to_starsapce)


    Xs = data_preprocessing(doc_embed_query)
    
    
    df_similarity = calculate_distance(Xs, embedding_labels, file_list, labels_l)
    
    
    
    
    df_similarity['filename'] = [file_list[i].split(",")[0] for i in range(len(file_list))] * len(labels_l)
    
    
    df_similarity = df_similarity[['filename', "file_label", "search_term", "score"]]

    # filter res by dist threshold
    
    df_similarity = df_similarity[df_similarity["score"] > threshold].reset_index(
        drop=True
    )


    df_similarity['score'] = 1 - df_similarity['score'] 
    

    df_similarity.to_csv(distance_file_path_rl, index=False)
            
    

    universe = pybedtools.BedTool(universe)

    file_list_db = meta_preprocessing(pd.read_csv(metadata_train), labels, files)
    file_list_query = meta_preprocessing(pd.read_csv(metadata_test), labels, files)
    


    distance_file_path_rr = os.path.join(
        output, "similarity_score_rr.csv"
    )

    doc_embed_dB = bed2vec(file_list_db, universe, input, "DB", temp_path, path_to_starsapce)

    db_vectors = data_preprocessing(doc_embed_dB)
   
    doc_embed_query = bed2vec(file_list_query, universe, input, "Query", temp_path, path_to_starsapce)

    query_vectors = data_preprocessing(doc_embed_query)


    df_similarity = calculate_distance_qc(
        db_vectors, query_vectors, file_list_db, file_list_query
    )

    df_similarity = df_similarity[["db_file", "test_file", "score"]]
    
    df_similarity['score'] = 1 - df_similarity['score'] 
    
    df_similarity.to_csv(distance_file_path_rr, index=False)
            


    