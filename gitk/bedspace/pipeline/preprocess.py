import logging

from ..const import PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)



def data_prepration(
        path_file_label: str, 
        univ: str
):
    path_file_label = path_file_label.split(",")
    path_file = path_file_label[0]
    labels = " ".join(
        ["__label__" + label for label in path_file_label[1:] if label != ""]
    )
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
            return [path_file, " ".join(list(file_regions["region"])) + " " + labels]
        except Exception:
            print("Error in reading file: ", path_file)
            return [path_file, " "]
    else:
        return [path_file, " "]
    

def meta_preprocessing(
    meta,
    labels,
    data_path
):
    cols = ["file_name"]
    cols.extend(labels)
    meta = meta[cols]
    meta["file_name"] = data_path + meta["file_name"]
    meta = meta.fillna("")
    meta[0] = meta.apply(",".join, axis=1)
    return meta[0]

def main(
    data_path: str,
    metadata: str,
    universe: str,
    output: str,
    labels: str,
):
    

    
    """
    Main function for the preprocess pipeline

    :param data_path: Path to bed files
    :param metadata: Path to metadata file
    :param universe: Path to universe file
    :param output: Path to output folder
    :param labels: Labels string (cell_type,target)
    """
    _LOGGER.info("Running preprocess...")

    # PLACE CODE FOR RUNNING PREPROCESS HERE
    
    meta_data = meta_preprocessing(pd.read_csv(metadata),labels, data_path)
    file_list = list(meta_data)
    trained_documents = []
    with Pool(n_process = 8) as p:
        trained_documents = p.starmap(
            data_prepration,
            [
                (x, universe)
                for x in file_list
            ],
        )
        p.close()
        p.join()
    
    print("Reading files done")

    df = pd.DataFrame(trained_documents, columns=["file_path", "context"])

    df = df.fillna(" ")
#     df = df[df.context != " "]

    with open(output, "w") as output_file:
        output_file.write("\n".join(df.context))
    output_file.close()
    
    
