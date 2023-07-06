import os
import gzip
import pickle
import yaml
from logging import getLogger
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import scanpy as sc
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from numba import config
from tqdm import tqdm

from .const import *
from .exceptions import *
from .utils import (
    LearningRateScheduler,
    ScheduleType,
    convert_anndata_to_documents,
    create_model_info_dict,
    remove_regions_below_min_count,
    shuffle_documents,
)

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"


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


class SCEmbed(Word2Vec):
    """
    Region2Vec model that extends the Word2Vec model from gensim.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        vector_size: int = DEFAULT_EMBEDDING_SIZE,
        min_count: int = DEFAULT_MIN_COUNT,
        threads: int = 1,
        seed: int = 42,
        callbacks: List[CallbackAny2Vec] = [],
        use_default_region_names: bool = True,
    ):
        """
        :param sc.AnnData data: The AnnData object containing the data to train on.
        :param int window_size: The size of the window to use for training.
        :param int vector_size: The size of the embedding vectors.
        :param int min_count: The minimum number of times a region must appear to be included in the model.
        :param int threads: The number of threads to use for training.
        :param int seed: The random seed to use for training.
        :param List[CallbackAny2Vec] callbacks: A list of callbacks to use for training.
        """
        self.callbacks = callbacks
        self.trained = False
        self.use_default_region_names = use_default_region_names
        self.region2vec = dict()

        # instantiate the Word2Vec model
        super().__init__(
            window=window_size,
            vector_size=vector_size,
            min_count=min_count,
            workers=threads,
            seed=seed,
            callbacks=callbacks,
        )

    def save_model(self, filepath: str, **kwargs):
        """
        Cant use the save function from gensim because it doesnt work on subclasses. see
        this issue: https://github.com/RaRe-Technologies/gensim/issues/1936

        Instead, we will just use pickle to dump the object.
        """
        with gzip.open(filepath, "wb") as f:
            pickle.dump(self, f)

    def export_model(
        self,
        out_path: str,
        model_export_name: str = DEFAULT_MODEL_EXPORT_FILE_NAME,
        universe_file_name: str = DEFAULT_UNIVERSE_EXPORT_FILE_NAME,
    ):
        """
        This function will do a full export of the model. This includes two files:
            1. the actual pickled `.model` file
            2. the universe `.bed` file that determines the universe of regions

        The result of this function can be directly uploaded to huggingface, to share with the world.

        :param str out_path: The path to the directory to save the model to.
        :param str model_export_name: The name of the model file to save - it is not recommended to change this.
        :param str universe_file_name: The name of the universe file to save - it is not recommended to change this.
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # save the model
        self.save_model(os.path.join(out_path, model_export_name))

        # save the universe to disk
        with open(os.path.join(out_path, universe_file_name), "w") as f:
            for region in self.region2vec:
                chr, start, end = region.split("_")
                f.write(f"{chr}\t{start}\t{end}\n")

    def load_model(self, filepath: str, **kwargs):
        """
        Cant use the load function from gensim because it doesnt work on subclasses. see
        this issue: https://github.com/RaRe-Technologies/gensim/issues/1936

        Instead we will just use pickle to load the object. Override the current object.
        """
        with gzip.open(filepath, "rb") as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)

    def train(
        self,
        data: Union[sc.AnnData, str],
        epochs: int = DEFAULT_EPOCHS,  # training cycles
        n_shuffles: int = DEAFULT_N_SHUFFLES,  # not the number of traiing cycles, actual shufle num
        gensim_epochs: Union[int, None] = DEFAULT_GENSIM_EPOCHS,
        report_loss: bool = True,
        lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        lr_schedule: Union[str, ScheduleType] = "linear",
    ):
        """
        Train the model. This is done in two steps: First, we shuffle the documents.
        Second, we train the model.

        :param int epochs: The number of epochs to train for (note: this is the number of times regions are shuffled, then fed to the model for training).
        :param int n_shuffles: The number of times to shuffle the regions within each document.
        :param int gensim_epochs: The number of epochs to train for within each shuffle (or main epoch).
        :param bool report_loss: Whether or not to report the loss after each epoch.
        :param float lr: The initial learning rate.
        :param float min_lr: The minimum learning rate.
        :param Union[str, ScheduleType] lr_schedule: The learning rate schedule to use.
        """
        # force to 1 for now (see: https://github.com/databio/gitk/pull/20#discussion_r1205683978)
        n_shuffles = 1

        if not isinstance(data, sc.AnnData) and not isinstance(data, str):
            raise TypeError(
                f"Data must be of type AnnData or str, not {type(data).__name__}"
            )

        # if the data is a string, assume it is a filepath
        if isinstance(data, str):
            data = sc.read_h5ad(data)

        if (
            not hasattr(data.var, CHR_KEY)
            or not hasattr(data.var, START_KEY)
            or not hasattr(data.var, END_KEY)
        ):
            _LOGGER.warn(
                "Data does not have `chr`, `start`, and `end` columns in the `var` attribute. Will fallback to default names"
            )

        # convert the data to a list of documents
        _LOGGER.info("Converting data to documents.")
        self.data = data
        # save anything in the `obs` attribute of the AnnData object
        # this lets users save any metadata they want
        # which can get mapped back to the embeddings
        self.obs = data.obs
        self.region_sets = convert_anndata_to_documents(
            data, self.use_default_region_names
        )

        # remove any regions that dont satisfy the min count
        _LOGGER.info("Removing regions that don't satisfy min count.")
        self.region_sets = remove_regions_below_min_count(
            self.region_sets, self.min_count
        )

        if report_loss:
            self.callbacks.append(ReportLossCallback())

        # create a learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            init_lr=lr, min_lr=min_lr, type=lr_schedule, n_epochs=epochs
        )

        # train the model using these shuffled documents
        _LOGGER.info("Training starting.")

        # build up the vocab
        super().build_vocab(
            self.region_sets,
            update=False if not self.trained else True,
            min_count=self.min_count,  # this shouldnt be needed but it is
        )

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
            self.region_sets = shuffle_documents(
                self.region_sets, n_shuffles=n_shuffles
            )

            # train the model on one iteration
            super().train(
                self.region_sets,
                total_examples=len(self.region_sets),
                epochs=1,  # for to 1 for now (see: https://github.com/databio/gitk/pull/20#discussion_r1205692089)
                callbacks=self.callbacks,
                compute_loss=report_loss,
                start_alpha=current_lr,
            )

            # update learning rates
            lr_scheduler.update()

            self.trained = True

        # once training is complete, create a region to vector mapping
        regions = list(self.wv.key_to_index.keys())

        # create a mapping from region to vector
        for word in regions:
            self.region2vec[word] = self.wv[word]

    def get_embedding(self, region: str) -> np.ndarray:
        """
        Get the embedding for a given region.

        :param str region: the region to get the embedding for
        """
        try:
            return self.wv[region]
        except KeyError:
            raise ValueError(f"Region {region} not found in model.")

    def get_embeddings(self, regions: List[str]) -> np.ndarray:
        """
        Get the embeddings for a list of regions.

        :param List[str] regions: the regions to get the embeddings for
        """
        return np.array([self.get_embedding(r) for r in regions])

    def region_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get the embeddings for each region in the original AnnData passed in.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        # return the key-value dict from the internal model
        # each key is a region and each value is the embedding
        return self.region2vec

    def cell_embeddings(self) -> sc.AnnData:
        """
        Get the cell embeddings for the original AnnData passed in. This should
        be called after training is complete. It is only useful for extracting the
        embeddings for the last chunk of data its seen.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        # get the embeddings for each cell
        cell_embeddings = []
        for cell in tqdm(self.region_sets, total=len(self.region_sets)):
            cell_embedding = np.mean(
                [self.get_embedding(r) for r in cell if r in self.region2vec], axis=0
            )
            cell_embeddings.append(cell_embedding)

        cell_embeddings = np.array(cell_embeddings)

        # attach embeddings to the AnnData object
        self.data.obsm["embedding"] = cell_embeddings
        return self.data

    def cell_embeddings(self) -> sc.AnnData:
        """
        Get the cell embeddings for the original AnnData passed in. This should
        be called after training is complete. It is only useful for extracting the
        embeddings for the last chunk of data its seen.
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        if self.data is None:
            raise ValueError(
                "Data not found. Please pass in an AnnData object to the constructor."
            )

        # get the embeddings for each cell
        cell_embeddings = []
        for cell in tqdm(self.region_sets, total=len(self.region_sets)):
            cell_embedding = np.mean(
                [self.get_embedding(r) for r in cell if r in self.region2vec], axis=0
            )
            cell_embeddings.append(cell_embedding)

        cell_embeddings = np.array(cell_embeddings)

        return cell_embeddings

    def attach_cell_embeddings(self, key_added: str = "embedding") -> sc.AnnData:
        """
        Get the cell embeddings for the original AnnData passed in. This should
        be called after training is complete. It is only useful for extracting the
        embeddings for the last chunk of data its seen.

        :param str key_added: the key to add the embeddings to in the obsm attribute of the AnnData object
        """
        if not self.trained:
            raise ValueError("Model has not been trained yet.")

        if not self.data:
            raise ValueError(
                "Data not found. Please pass in an AnnData object to the constructor."
            )

        if not self.region_sets:
            self.region_sets = convert_anndata_to_documents(self.data)

        # get the embeddings for each cell
        cell_embeddings = []
        for cell in tqdm(self.region_sets, total=len(self.region_sets)):
            cell_embedding = np.mean(
                [self.get_embedding(r) for r in cell if r in self.region2vec], axis=0
            )
            cell_embeddings.append(cell_embedding)

        cell_embeddings = np.array(cell_embeddings)

        # attach embeddings to the AnnData object
        self.data.obsm[key_added] = cell_embeddings
        return self.data

    def embeddings_to_csv(self, output_path: str):
        """
        Save the embeddings to a CSV file.

        :param str output_path: the path to save the embeddings to
        """
        embeddings = self.cell_embeddings()
        embeddings_df = embeddings.obs

        # split the embeddings for each barcode into each dimension and save them
        parsed_rows = []
        for row in list(embeddings_df.iterrows()):
            row_dict = row[1].to_dict()
            new_row_dict = {
                f"embedding_dim_{i}": row_dict["embedding"][i]
                for i in range(self.vector_size)
            }
            if "id" in row_dict:
                new_row_dict["id"] = row_dict["id"]
            parsed_rows.append(new_row_dict)

        parsed_df = pd.DataFrame(parsed_rows)
        parsed_df.to_csv(output_path, index=False)

    def anndata_to_embeddings(self, adata: sc.AnnData) -> sc.AnnData:
        """
        This function will take an AnnData object and add a column to the obs
        attribute of the AnnData object that contains the embeddings for each
        cell.

        It does this by taking the regions for each cell and averaging the
        embeddings for each region.

        :param sc.AnnData adata: the AnnData object to add the embeddings to
        :return sc.AnnData: the AnnData object with the embeddings added
        """
        region_sets = convert_anndata_to_documents(adata, self.use_default_region_names)
        cell_embeddings = []
        for cell in tqdm(region_sets, total=len(region_sets)):
            cell_embedding = np.mean([self.get_embedding(r) for r in cell], axis=0)
            cell_embeddings.append(cell_embedding)

        # attach embeddings to the AnnData object
        self.data.obs["embedding"] = cell_embeddings
        return self.data

    def __call__(self, region: str) -> np.ndarray:
        """
        Get the embedding for a given region.

        :param str region: the region to get the embedding for
        """
        return self.get_embedding(region)
