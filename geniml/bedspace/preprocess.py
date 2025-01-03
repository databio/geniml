import logging
from multiprocessing import Pool
import pandas as pd
from geniml.tokenization.main import TreeTokenizer
import datetime
from tqdm import tqdm

from .helpers import meta_preprocessing, data_preparation
from .const import PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def main(
    data_path: str,
    metadata: str,
    universe: str,
    output: str,
    labels: str
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
    _LOGGER.info(f"Start: {datetime.datetime.now()}")


    # PLACE CODE FOR RUNNING PREPROCESS HERE
    universe = TreeTokenizer(universe)
    file_list = meta_preprocessing(metadata, labels, data_path, "train")
    trained_documents = []
    with Pool(processes=8) as p:
        trained_documents = tqdm(
            p.starmap(
                data_preparation, 
                [(x, universe, "train") for x in file_list]
                ), 
                total=len(file_list)
            )
        p.close()
        p.join()

    print("Reading files done")

    df = pd.DataFrame(trained_documents, columns=["file_path", "context"])
    df = df.fillna(" ")

    with open(f"{output}train_input.txt", "w") as output_file:
        output_file.write("\n".join(df.context))
    output_file.close()

    _LOGGER.info("Traning sample preprocess done.")
    _LOGGER.info(f"End: {datetime.datetime.now()}")


