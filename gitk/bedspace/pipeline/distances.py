import logging

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
    df_distance_matrix["file_label"] = y_files
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


def meta_preprocessing(meta, labels, data_path):
    cols = ["file_name"]
    cols.extend(labels)
    meta = meta[cols]
    meta["file_name"] = data_path + meta["file_name"]
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


    # Data prepration
    tokenized_documents = []
    with Pool(n_process) as p:
        tokenized_documents = p.starmap(
            data_prepration_test, [(x, universe) for x in file_list]
        )
        p.close()
        p.join()
    print("Reading files done")

    df = pd.DataFrame(tokenized_documents, columns=["file_path", "context"])
    df = df.fillna(" ")
#     df = df[df.context != " "]

    with open(docs_test, "w") as file:
        file.write("\n".join(df.context))
    file.close()


    print("Writing files done")

    starspace_path = os.path.join(
        path_to_starsapce,
        "embed_doc",
    )

    output = check_output([starspace_path, input, docs_test]).decode("utf-8")

    with open(doc_embed_test, "w") as out:
        out.write(output)
        out.close()

    Xs = data_preprocessing(doc_embed_test)

    for i in range(len(file_list)):
        y = ",".join(file_list[i].split(",")[1:])
        X = Xs[i]

        df_similarity = calculate_distance(X, embedding_labels, y, labels)

        df_similarity['filename'] = [file_list[i].split(",")[0]] * len(labels)


        df_similarity = df_similarity[['filename', "file_label", "search_term", "score"]]

        # filter res by dist threshold
        thresh = 0.5
        df_similarity = df_similarity[df_similarity["score"] < thresh].reset_index(
            drop=True
        )


        if os.path.exists(distance_file_path):
            df_similarity.to_csv(distance_file_path, header=False, index=None, mode="a")
        else:
            df_similarity.to_csv(distance_file_path, index=False)
            
            
            
            
            
    
    universe = pybedtools.BedTool(args.univ_path)

    meta_data_db, assembly, file_list_db = meta(args.meta_path_db)
    meta_data_query, assembly, file_list_query = meta(args.meta_path_query)
    


    # define file path
    model = os.path.join(args.output_path, "starspace_model_{}".format(assembly))


    dist = os.path.join(
        args.output_path, "query_db_similarity_score_{}.csv".format(assembly)
    )


    doc_embed_dB = bed2vec(file_list_db, universe, model, assembly, "DB", args.output_path)

    db_vectors = data_preprocessing(doc_embed_dB)

    print(db_vectors.shape)
    doc_embed_query = bed2vec(file_list_query, universe, model, assembly, "Query", args.output_path)

    query_vectors = data_preprocessing(doc_embed_query)

    print(query_vectors.shape)
    
    for i in range(len(query_vectors)):

        query_vector = query_vectors[i]

        df_similarity = calculate_distance(
            db_vectors, query_vectors, file_list_db, file_list_query
        )

        df_similarity = df_similarity[["db_file", "test_file", "score"]]

        if os.path.exists(dist):
            df_similarity.to_csv(dist, header=False, index=None, mode="a")
        else:
            df_similarity.to_csv(dist, index=False)
            


    
    
    
