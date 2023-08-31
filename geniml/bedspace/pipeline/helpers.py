import os
from hashlib import md5
from multiprocessing import Pool
from subprocess import check_output

import pandas as pd
import pybedtools


def data_prepration(path_file_label: str, univ: str):
    path_file_label = path_file_label.split(",")
    path_file = path_file_label[0]
    labels = " ".join(["__label__" + label for label in path_file_label[1:] if label != ""])
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
            return [
                path_file,
                " ".join(list(file_regions["region"])) + " " + labels,
            ]
        except Exception:
            print("Error in reading file: ", path_file)
            return [path_file, " "]
    else:
        return [path_file, " "]


def data_prepration_test(path_file_label, univ):
    #     print(path_file_label)
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


def bed2vec(file_list, universe, model, assembly, source, output_path):
    docs = os.path.join(output_path, "documents_{}.txt".format(assembly))
    files = os.path.join(output_path, "filenames_{}.txt".format(assembly))

    doc_embed = os.path.join(
        output_path,
        "{}_starspace_embed_{}_{}.txt".format(source, assembly, source),
    )

    documents = []
    with Pool(16) as p:
        documents = p.starmap(data_prepration_test, [(x, universe) for x in file_list])
        p.close()
        p.join()
    print("Reading files done")

    df = pd.DataFrame(documents, columns=["file_path", "context"])
    df = df.fillna(" ")
    df = df[df.context != " "]

    with open(docs, "w") as input_file:
        input_file.write("\n".join(df.context))
    input_file.close()

    #     with open(files, "w") as input_file:
    #         input_file.write("\n".join(df.file_path))
    #     input_file.close()

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

    return doc_embed
