import scanpy as sc
import pandas as pd
import numpy as np

from typing import Dict, List, Union
from concurrent.futures import ThreadPoolExecutor
from random import shuffle
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from numba import config
from logging import getLogger
from tqdm import tqdm

from .const import *
from .utils import LearningRateScheduler, ScheduleType

_LOGGER = getLogger(PKG_NAME)

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"


# shuffle the document to generate data for word2vec
def shuffle_documents(
    documents: List[List[str]],
    n_shuffles: int,
    threads: int = None,
) -> List[List[str]]:
    """
    Shuffle around the genomic regions for each cell to generate a "context".

    :param List[List[str]] documents: the document list to shuffle.
    :param int n_shuffles: The number of shuffles to conduct.
    :param int threads: The number of threads to use for shuffling.
    """

    def shuffle_list(l: List[str], n: int) -> List[str]:
        for _ in range(n):
            shuffle(l)
        return l

    _LOGGER.debug(f"Shuffling documents {n_shuffles} times.")
    shuffled_documents = documents.copy()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        shuffled_documents = list(
            tqdm(
                executor.map(
                    shuffle_list, shuffled_documents, [n_shuffles] * len(documents)
                ),
                total=len(documents),
            )
        )
    return shuffled_documents


class ReportLossCallback(CallbackAny2Vec):
    """
    Callback to report loss after each epoch.
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model: Word2Vec):
        loss = model.get_latest_training_loss()
        _LOGGER.info(f"Epoch {self.epoch} complete. Loss: {loss}")
        self.epoch += 1


class Region2Vec(Word2Vec):
    """
    Region2Vec model that extends the Word2Vec model from gensim.
    """

    def __init__(
        self,
        data: sc.AnnData,
        window_size: int = DEFAULT_WINDOW_SIZE,
        vector_size: int = DEFAULT_EMBEDDING_SIZE,
        min_count: int = 10,
        threads: int = 1,
        seed: int = 42,
        callbacks: List[CallbackAny2Vec] = [],
    ):
        # convert the data to a list of documents
        _LOGGER.info("Converting data to documents.")
        self.data = data
        self.region_sets = convert_anndata_to_documents(data)
        self.callbacks = callbacks

        # save anything in the `obs` attribute of the AnnData object
        # this lets users save any metadata they want
        # which can get mapped back to the embeddings
        self.obs = data.obs
        self.trained = False

        # instantiate the Word2Vec model
        super().__init__(
            window=window_size,
            vector_size=vector_size,
            min_count=min_count,
            workers=threads,
            seed=seed,
            callbacks=callbacks,
        )

    def train(
        self,
        epochs: int = DEFAULT_EPOCHS, # training cycles
        n_shuffles: int = DEAFULT_N_SHUFFLES, # not the number of traiing cycles, actual shufle num
        gensim_epochs: Union[int, None] = DEFAULT_GENSIM_EPOCHS,
        report_loss: bool = True,
        lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        lr_schedule: Union[str, ScheduleType] = "linear",
    ):
        """
        Train the model. This is done in two steps: First, we shuffle the documents.
        Second, we train the model.
        """
        self.trained = True
        if report_loss:
            self.callbacks.append(ReportLossCallback())

        lr_scheduler = LearningRateScheduler(
            init_lr=lr, min_lr=min_lr, type=lr_schedule, n_epochs=epochs
        )

        # train the model using these shuffled documents
        _LOGGER.info("Training starting.")

        for shuffle_num in range(epochs):
            # update current values
            current_lr = lr_scheduler.get_lr()
            current_loss = self.get_latest_training_loss()

            # update user
            _LOGGER.info(
                f"SHUFFLE {shuffle_num} - lr: {current_lr}, loss: {current_loss}"
            )
            _LOGGER.info("Shuffling documents.")

            # shuffle regions
            self.region_sets = shuffle_documents(self.region_sets, n_shuffles=n_shuffles)

            # update vocab and train
            super().build_vocab(self.region_sets, update=True if shuffle_num > 0 else False)
            super().train(
                self.region_sets,
                total_examples=len(self.region_sets),
                epochs=gensim_epochs or 1,  # use the epochs passed in or just one
                callbacks=self.callbacks,
                compute_loss=report_loss,
                start_alpha=current_lr,
            )
        
        # once training is complete, create a region to vector mapping
        regions = list(self.wv.key_to_index.keys())
        self.region2vec = {word: self.wv[word] for word in regions}

    def get_embedding(self, region: str) -> np.ndarray:
        """
        Get the embedding for a given region.

        :param str region: the region to get the embedding for
        """
        return self.wv[region]

    def get_embeddings(self, regions: List[str]) -> np.ndarray:
        """
        Get the embeddings for a list of regions.

        :param List[str] regions: the regions to get the embeddings for
        """
        return np.array([self.get_embedding(r) for r in regions])

    def cell_embeddings(self) -> sc.AnnData:
        """
        Get the cell embeddings for the original AnnData passed in.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        # get the embeddings for each cell
        cell_embeddings = []
        for cell in tqdm(self.region_sets, total=len(self.region_sets)):
            cell_embedding = np.mean([self.get_embedding(r) for r in cell], axis=0)
            cell_embeddings.append(cell_embedding)

        # attach embeddings to the AnnData object
        self.data.obs["embedding"] = cell_embeddings
        return self.data


def load_scanpy_data(path_to_h5ad: str) -> sc.AnnData:
    """
    Load in the h5ad file that holds all of the information
    for our single-cell data with scanpy.

    :param str path_to_h5ad: the path to the h5ad file made with scanpy
    """
    return sc.read_h5ad(path_to_h5ad)


def extract_region_list(region_df: pd.DataFrame) -> List[str]:
    """
    Parse the `var` attribute of the scanpy.AnnData object and
    return a list of regions from the matrix

    :param pandas.DataFrame region_df: the regions dataframe to parse
    """
    _LOGGER.info("Extracting region list from matrix.")
    regions_parsed = []
    for r in tqdm(region_df.iterrows(), total=region_df.shape[0]):
        r_dict = r[1].to_dict()
        regions_parsed.append(
            " ".join([r_dict["chr"], str(r_dict["start"]), str(r_dict["end"])])
        )
    return regions_parsed


def remove_zero_regions(cell_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Removes any key-value pairs in a dictionary where the value (copy number)
    is equal to zero (no signal). This is done using dictionary comprehension
    as it is much faster.

    :param cell_dict Dict[str, int]: the cell dictionary with region index keys and copy number values
    """
    return {k: v for k, v in cell_dict.items() if v > 0}


def convert_anndata_to_documents(anndata: sc.AnnData) -> List[List[str]]:
    """
    Parses the scanpy.AnnData object to create the required "documents" object for
    training the Word2Vec model. Each row (or cell) is treated as a "document". That
    is, each region is a "word", and the total collection of regions is the "document".

    :param scanpy.AnnData anndata: the AnnData object to parse.
    """
    regions_parsed = extract_region_list(anndata.var)
    sc_df = anndata.to_df()
    docs = []
    _LOGGER.info("Generating documents.")

    for row in tqdm(sc_df.iterrows(), total=sc_df.shape[0]):
        row_dict = row[1].to_dict()
        row_dict = remove_zero_regions(row_dict)
        new_doc = []
        for region_indx in row_dict:
            region_str = regions_parsed[int(region_indx)]
            # replace spaces with underscores
            region_str = region_str.replace(" ", "_")
            new_doc.append(region_str)
        docs.append(new_doc)
    return docs
