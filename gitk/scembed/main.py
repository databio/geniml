import gzip
import os
import pickle
from logging import getLogger
from typing import List, Union
from multiprocessing import cpu_count

import numpy as np
import scanpy as sc
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from numba import config

from ..io import Region, RegionSet
from ..tokenization import InMemTokenizer
from .const import *
from .exceptions import *
from .utils import (
    LearningRateScheduler,
    ScheduleType,
    convert_anndata_to_documents,
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

        :param str filepath: The path to save the model to.
        :param kwargs: Additional arguments to pass to pickle.dump
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

        :param str filepath: The path to load the model from.
        :param kwargs: Additional arguments to pass to pickle.load
        """
        with gzip.open(filepath, "rb") as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)

    def train(
        self,
        data: Union[sc.AnnData, str],
        epochs: int = DEFAULT_EPOCHS,  # training cycles
        n_shuffles: int = DEFAULT_N_SHUFFLES,  # not the number of traiing cycles, actual shufle num
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
            raise TypeError(f"Data must be of type AnnData or str, not {type(data).__name__}")

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
        self.region_sets = convert_anndata_to_documents(data, self.use_default_region_names)

        # remove any regions that dont satisfy the min count
        _LOGGER.info("Removing regions that don't satisfy min count.")
        self.region_sets = remove_regions_below_min_count(self.region_sets, self.min_count)

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
            _LOGGER.info(f"SHUFFLE {shuffle_num} - lr: {current_lr}, loss: {current_loss}")
            _LOGGER.info("Shuffling documents.")

            # shuffle regions
            self.region_sets = shuffle_documents(self.region_sets, n_shuffles=n_shuffles)

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

    def __call__(self, region: str) -> np.ndarray:
        """
        Get the embedding for a given region.

        :param str region: the region to get the embedding for
        """
        return self.get_embedding(region)


#
# BEGIN V2 of SCEmbed class
#
class SCEmbedV2(Word2Vec):
    def __init__(
        self,
        model_path: str = None,
        tokenizer: InMemTokenizer = None,
        universe: Union[RegionSet, str] = None,
        **kwargs,
    ):
        """
        The SCEmbedV2 class is a subclass of the Word2Vec class from gensim. It is a
        Region2Vec model that extends the Word2Vec model from gensim.

        :param str model_path: The path to a pretrained model to load.
        :param InMemTokenizer tokenizer: The tokenizer to use for tokenizing region sets.
        :param Universe universe: The universe to use for tokenizing region sets.
        :param kwargs: Additional arguments to pass to the Word2Vec constructor.
        """
        # if model_path is given, download a pretrained model
        # from the HuggingFace Hub
        if model_path:
            self._init_from_huggingface(model_path)
        else:
            # otherwise, initialize a new model, with a new universe and tokenizer
            # since the model is untrained, we have no vocabulary yet,
            # and thus need a fresh tokenizer and universe
            self.tokenizer = tokenizer or InMemTokenizer()

        # set other attributes
        self.callbacks = kwargs.get("callbacks") or []
        self.trained = False
        self.region2vec = dict()

        # instantiate the Word2Vec model
        super().__init__(
            window=kwargs.get("window_size") or DEFAULT_WINDOW_SIZE,
            vector_size=kwargs.get("vector_size") or DEFAULT_EMBEDDING_SIZE,
            min_count=kwargs.get("min_count") or DEFAULT_MIN_COUNT,
            workers=kwargs.get("threads") or cpu_count() - 2,
            seed=kwargs.get("seed") or 42,  #
            callbacks=kwargs.get("callbacks") or [],
        )

    def _validate_data(self, data: Union[sc.AnnData, str]) -> sc.AnnData:
        """
        Validate the data is of the correct type and has the required columns.

        :param sc.AnnData | str data: The AnnData object containing the data to train on (can be path to AnnData).

        :return sc.AnnData: The AnnData object.
        """
        if not isinstance(data, sc.AnnData) and not isinstance(data, str):
            raise TypeError(f"Data must be of type AnnData or str, not {type(data).__name__}")

        # if the data is a string, assume it is a filepath
        if isinstance(data, str):
            data = sc.read_h5ad(data)

        # validate the data has the required columns
        if (
            not hasattr(data.var, CHR_KEY)
            or not hasattr(data.var, START_KEY)
            or not hasattr(data.var, END_KEY)
        ):
            raise ValueError(
                "Data does not have `chr`, `start`, and `end` columns in the `var` attribute. This is required."
            )

        return data

    def train(
        self,
        data: Union[sc.AnnData, str],
        epochs: int = DEFAULT_EPOCHS,  # training cycles
        report_loss: bool = True,
        lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        lr_schedule: Union[str, ScheduleType] = "linear",
    ):
        """
        Train the model. This is done in two steps: First, we shuffle the documents.
        Second, we train the model.

        :param sc.AnnData | str data: The AnnData object containing the data to train on (can be path to AnnData).
        :param int epochs: The number of epochs to train for (note: this is the number of times regions are shuffled, then fed to the model for training).
        :param bool report_loss: Whether or not to report the loss after each epoch.
        :param float lr: The initial learning rate.
        :param float min_lr: The minimum learning rate.
        :param Union[str, ScheduleType] lr_schedule: The learning rate schedule to use.
        """
        # force to 1 for now (see: https://github.com/databio/gitk/pull/20#discussion_r1205683978)
        n_shuffles = 1

        # validate the data coming in
        data = self._validate_data(data)

        _LOGGER.info("Tokenizing.")

        # extract out the chr, start, end columns
        chrs = data.var[CHR_KEY].values.tolist()
        starts = data.var[START_KEY].values.tolist()
        ends = data.var[END_KEY].values.tolist()
        regions = [Region(c, int(s), int(e)) for c, s, e in zip(chrs, starts, ends)]

        # fit the tokenizer on the regions
        self.tokenizer.fit(regions)

        # convert the data to a list of documents
        region_sets = self.tokenizer.tokenize(data)

        # convert list of lists of regions to list of list of strings of form chr_start_end
        region_sets = [
            [r.chr + "_" + str(r.start) + "_" + str(r.end) for r in region_set]
            for region_set in region_sets
        ]

        if report_loss:
            self.callbacks.append(ReportLossCallback())

        # create a learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            init_lr=lr, min_lr=min_lr, type=lr_schedule, n_epochs=epochs
        )

        # train the model using these shuffled documents
        _LOGGER.info("Training begin.")

        # build up the vocab
        super().build_vocab(
            region_sets,
            update=False if not self.trained else True,
            min_count=self.min_count,  # this shouldnt be needed but it is
        )

        for shuffle_num in range(epochs):
            # update current values
            current_lr = lr_scheduler.get_lr()
            current_loss = self.get_latest_training_loss()

            # update user
            _LOGGER.info(f"SHUFFLE {shuffle_num} - lr: {current_lr}, loss: {current_loss}")
            _LOGGER.info("Shuffling documents.")

            # shuffle regions
            region_sets = shuffle_documents(region_sets, n_shuffles=n_shuffles)

            # train the model on one iteration
            super().train(
                region_sets,
                total_examples=len(region_sets),
                epochs=1,  # for to 1 for now (see: https://github.com/databio/gitk/pull/20#discussion_r1205692089)
                callbacks=self.callbacks,
                compute_loss=report_loss,
                start_alpha=current_lr,
            )

            # update learning rates
            lr_scheduler.update()

            self.trained = True

        # once training is complete, create a region to vector mapping
        learned_regions = list(self.wv.key_to_index.keys())

        # create a mapping from region to vector
        for word in learned_regions:
            self.region2vec[word] = self.wv[word]

        def __call__(self):
            pass
