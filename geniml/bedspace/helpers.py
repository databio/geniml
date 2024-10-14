import os
import numpy as np
import pandas as pd
import pybedtools
from multiprocessing import Pool
from subprocess import check_output
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def meta_preprocessing(meta_path, labels, input_path, mode, chunksize=10000):
    """
    Process the metadata file in chunks, combining file paths with associated labels.
    
    Parameters:
    - meta_path (str): Path to the metadata CSV containing file names and labels.
    - labels (list): List of column names representing labels to concatenate.
    - input_path (str): Directory path to prepend to each file name.
    - chunksize (int): Number of rows to process at a time (default: 10000).
    
    Returns:
    - file_list (df): List of combined file paths and labels
    """
    if isinstance(labels, str):
        labels = labels.split(",")

    # Define the columns to use
    cols = ["file_name"]
    if mode == "train":
        cols.extend(labels)

    file_list = []
    
    # Read the CSV file in chunks
    for chunk in pd.read_csv(meta_path, usecols=cols, chunksize=chunksize):
        chunk["file_name"] = input_path + chunk["file_name"]
        chunk = chunk.fillna("")
        chunk[0] = chunk.apply(",".join, axis=1)
        file_list.extend(chunk[0].tolist())
    
    return file_list


def data_preparation(path_file_label: str, univ: str, mode: str):
    """
    Convert input region set data (BED files) into a StarSpace acceptable format.

    Parameters:
    - path_file_label (str): path to the BED file.
    - univ(bedtool obj): universe to intersect the BED file with.
    - mode (str): either "train" or "test".

    Returns:
    - (list): A list containing the file path and the formatted region set data (with or without labels).
    """

    # Split the input string into the file path and labels
    path_file_label = path_file_label.split(",")
    path_file = path_file_label[0]
    labels = ""

    # Generate labels only for 'train' type
    if mode == "train":
        labels = " ".join(["__label__" + label for label in path_file_label[1:] if label != ""])
    
    # Check if the file exists before processing
    if not os.path.exists(path_file):
        return [path_file, " "]

    try:
        # Read the BED file and intersect with the universal set
        df = pybedtools.BedTool(path_file)
        file_regions = univ.intersect(df, wa=True, nonamecheck=True)
        file_regions.columns = ["chrom", "start", "end"]
        # Return empty string if no regions found
        if len(file_regions) == 0:
            return [path_file, " "]
        # Convert to DataFrame and remove duplicates
        file_regions = file_regions.to_dataframe().drop_duplicates()
        # Create the "region" column by combining chrom, start, and end
        file_regions["region"] = (
            file_regions["chrom"]
            + "_"
            + file_regions["start"].astype(str)
            + "_"
            + file_regions["end"].astype(str)
        )
        # Build the final result based on whether it's 'train' or not
        regions_str = " ".join(list(file_regions["region"]))
        if mode == "train":
            return [path_file, regions_str + " " + labels]
        else: 
            return [path_file, regions_str]
    # Print error details for debugging
    except Exception:
        print("Error in reading file: ", path_file)
        return [path_file, " "]


def get_label_embedding(path_word_embedding, label_prefix):
    """
    Extract label embeddings from the trained model's word embeddings.

    Parameters:
    - path_word_embedding (str): Path to the word embedding file.
    - label_prefix (str): Prefix used to identify label embeddings.

    Returns:
    - label_vectors (list): List of label embedding vectors.
    - labels (list): List of labels without the prefix.
    """

    labels = []
    label_vectors = []
    # Read the word embeddings
    word_embedding = pd.read_csv(path_word_embedding, sep="\t", header=None)
    # Filter rows that contain the label prefix
    vectors = word_embedding[word_embedding[0].str.contains(label_prefix)]  # .reset_index()
    # Extract label vectors and labels
    for l in range(len(vectors)):
        label_vectors.append((list(vectors.iloc[l])[1:]))
        labels.append(list(vectors.iloc[l])[0].replace(label_prefix, ""))
    return label_vectors, labels


def bed2vec(file_list, univ, model, docs, doc_embed, path_to_starsapce):
    """
    Predict sample (region set) embedding using StarSpace. Write output to a file.
    
    Parameters:
    - file_list (df): List of BED files to process.
    - univ (bedtool obj): universe for intersecting with regions.
    - model (str): Path to the trained StarSpace model.
    - docs (str): Path to save the context documents for embedding.
    - doc_embed (str): Path to save the predicted document embeddings.
    - path_to_starspace (str): Directory where StarSpace is located.
    """

    trained_documents = []
    with Pool(16) as p:
        trained_documents = tqdm(p.starmap(data_preparation, [(x, univ, "test") for x in file_list]), total=len(file_list))
        p.close()
        p.join()
    print("Reading files done")
    df = pd.DataFrame(trained_documents, columns=["file_path", "context"])
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


def get_embedding_matrix(path_embeded_document):
    """
    Create sample and label embedding matrix from the given document embedding file.
    
    Parameters:
    - path_embeded_document (str): Path to the document embedding file (CSV format).
    
    Returns:
    - Xs (list): A list of embedding vectors (samples) in the form of floats.
    """

    # Read the document embedding file
    document_embedding = pd.read_csv(path_embeded_document, header=None)
    # Split on "__label__", shift the data, and clean up
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


def calculate_distance(X_files, X_labels, y_files, y_labels, mode):
    """
    Calculate the cosine distance matrix between two sets of file embeddings 
    and return a melted DataFrame with the distances and associated labels.
    
    Parameters:
    - X_files (array-like): test embeddings.
    - X_labels (array-like): train embeddings.
    - y_files (list): test file name.
    - y_labels (list): trained label or training file names.
    - mode (str): Specifies the processing type; "rl" or "rr".
    
    Returns:
    - file_distance (df): A melted DataFrame containing distances and labels.
    """

    # Convert inputs to numpy arrays
    X_files = np.array(X_files)
    X_labels = np.array(X_labels)
    
    # Calculate the cosine distance matrix
    distance_matrix = distance.cdist(X_files, X_labels, "cosine")
    df_distance_matrix = pd.DataFrame(distance_matrix)
    df_distance_matrix.columns = y_labels
    
    # Handling different types
    if mode == "rl":
        df_distance_matrix["file_label"] = [y_files[i].split(",")[1] for i in range(len(y_files))]
        file_distance = pd.melt(
            df_distance_matrix,
            id_vars="file_label",
            var_name="search_term",
            value_name="score"
        )
    elif mode == "rr":
        df_distance_matrix["db_file"] = y_files
        file_distance = pd.melt(
            df_distance_matrix,
            id_vars="db_file",
            var_name="test_file",
            value_name="score"
        )
        # Apply MinMaxScaler to normalize the 'score' column
        scaler = MinMaxScaler()
        file_distance["score"] = scaler.fit_transform(np.array(file_distance["score"]).reshape(-1, 1))


    return file_distance
