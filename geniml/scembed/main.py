import os
from logging import getLogger
from typing import Union

import numpy as np
import scanpy as sc
import torch
from huggingface_hub import hf_hub_download
from rich.progress import track
from yaml import safe_load, safe_dump

from ..region2vec.utils import LearningRateScheduler, shuffle_documents
from ..region2vec.main import Region2Vec
from ..tokenization import ITTokenizer, Tokenizer
from ..region2vec.const import (
    POOLING_TYPES,
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_COUNT,
    DEFAULT_EPOCHS,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
    CONFIG_FILE_NAME,
)

from .const import (
    CHR_KEY,
    END_KEY,
    START_KEY,
    MODULE_NAME,
)

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")


class ScEmbed:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: ITTokenizer = None,
        device: str = None,
        **kwargs,
    ):
        """
        Initialize ScEmbed.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param embedding_dim: Dimension of the embedding.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        super().__init__()
        self.model_path: str = model_path
        self.tokenizer: ITTokenizer = tokenizer
        self.trained: bool = False
        self._model: Region2Vec = None

        if model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True

        elif tokenizer is not None:
            self._init_model(**kwargs)

        # set the device
        self._target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _init_model(self, **kwargs):
        """
        Initialize the core model. This will initialize the model from scratch.

        :param kwargs: Additional keyword arguments to pass to the model.
        """
        if self.tokenizer:
            self._vocab_length = len(self.tokenizer)
            self._model = Region2Vec(
                len(self.tokenizer),
                embedding_dim=kwargs.get("embedding_dim", DEFAULT_EMBEDDING_SIZE),
            )

    @property
    def model(self):
        """
        Get the core Region2Vec model.
        """
        return self._model

    def add_tokenizer(self, tokenizer: Tokenizer, **kwargs):
        """
        Add a tokenizer to the model. This should be use when the model
        is not initialized with a tokenizer.

        :param tokenizer: Tokenizer to add to the model.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        if self._model is not None:
            raise RuntimeError("Cannot add a tokenizer to a model that is already initialized.")

        self.tokenizer = tokenizer
        if not self.trained:
            self._init_model(**kwargs)

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        self._model_path = model_path
        self._universe_path = vocab_path

        # init the tokenizer - only one option for now
        self.tokenizer = ITTokenizer(vocab_path)

        # load the model state dict (weights)
        params = torch.load(model_path)

        # get the model config (vocab size, embedding size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self._model = Region2Vec(
            config["vocab_size"],
            embedding_dim=config["embedding_size"],
        )
        self._model.load_state_dict(params)

    def _init_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those. This
        includes both the actual model and the tokenizer.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param kwargs: Additional keyword arguments to pass to the hf download function.
        """
        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self._load_local_model(model_file_path, universe_path, config_path)

    @classmethod
    def from_pretrained(
        cls,
        path_to_files: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
    ) -> "ScEmbed":
        """
        Load the model from a set of files that were exported using the export function.

        :param str path_to_files: Path to the directory containing the files.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        """
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(model_file_path, universe_file_path, config_file_path)
        instance.trained = True

        return instance

    def _validate_data_for_training(self, data: Union[sc.AnnData, str]) -> sc.AnnData:
        """
        Validate the data is of the correct type and has the required columns

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
        window_size: int = DEFAULT_WINDOW_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        min_count: int = DEFAULT_MIN_COUNT,
        num_cpus: int = 1,
        seed: int = 42,
        checkpoint_path: str = MODEL_FILE_NAME,
        save_model: bool = False,
        gensim_params: dict = {},
    ):
        """
        Train the model.

        :param sc.AnnData data: The AnnData object containing the data to train on (can be path to AnnData).
        :param int window_size: The window size to use for training.
        :param int epochs: The number of epochs to train for.
        :param int min_count: The minimum count for a region to be included in the vocabulary.
        :param int num_cpus: The number of cpus to use for training.
        :param int seed: The seed to use for training.
        :param str checkpoint_path: The path to save the model to.
        :param bool save_model: Whether to save the model after training.
        :param dict gensim_params: Additional keyword arguments to pass to the gensim model.

        :return np.ndarray: The losses for each epoch.
        """
        from gensim.models import Word2Vec as GensimWord2Vec

        # validate a model exists
        if self._model is None:
            raise RuntimeError(
                "Cannot train a model that has not been initialized. Please initialize the model first using a tokenizer or from a huggingface model."
            )

        _LOGGER.info("Validating data.")
        data = self._validate_data_for_training(data)

        # create gensim model that will be used to train
        _LOGGER.info("Creating gensim model.")
        gensim_model = GensimWord2Vec(
            vector_size=self._model.embedding_dim,
            window=window_size,
            min_count=min_count,
            workers=num_cpus,
            seed=seed,
            **gensim_params,
        )

        # convert to tokens for training
        tokenized_data = self.tokenizer.tokenize(data)

        _LOGGER.info("Building vocabulary.")
        tokenized_data = [
            [str(t.id) for t in region_set]
            for region_set in track(
                tokenized_data,
                total=len(tokenized_data),
                description="Converting to strings.",
            )
        ]
        gensim_model.build_vocab(tokenized_data)

        # create our own learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            n_epochs=epochs,
        )

        # train the model
        losses = []

        for epoch in track(range(epochs), description="Training model", total=epochs):
            # shuffle the data
            _LOGGER.info(f"Starting epoch {epoch+1}.")
            _LOGGER.info("Shuffling data.")
            shuffled_data = shuffle_documents(
                tokenized_data, n_shuffles=1
            )  # shuffle once per epoch, no need to shuffle more
            gensim_model.train(
                shuffled_data,
                epochs=1,  # train for 1 epoch at a time, shuffle data each time
                compute_loss=True,
                total_words=gensim_model.corpus_total_words,
            )

            # log out and store loss
            _LOGGER.info(f"Loss: {gensim_model.get_latest_training_loss()}")
            losses.append(gensim_model.get_latest_training_loss())

            # update the learning rate
            lr_scheduler.update()

        # once done training, set the weights of the pytorch model in self._model
        for id in track(
            gensim_model.wv.key_to_index,
            total=len(gensim_model.wv.key_to_index),
            description="Setting weights.",
        ):
            self._model.projection.weight.data[int(id)] = torch.tensor(gensim_model.wv[id])

        # set the model as trained
        self.trained = True

        # export
        if save_model:
            self.export(checkpoint_path)

        return np.array(losses)

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        universe_file: str = UNIVERSE_FILE_NAME,
        config_file: str = CONFIG_FILE_NAME,
    ):
        """
        Function to facilitate exporting the model in a way that can
        be directly uploaded to huggingface. This exports the model
        weights and the vocabulary.

        :param str path: Path to export the model to.
        """
        # make sure the model is trained
        if not self.trained:
            raise RuntimeError("Cannot export an untrained model.")

        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # export the model weights
        torch.save(self._model.state_dict(), os.path.join(path, checkpoint_file))

        # export the vocabulary
        with open(os.path.join(path, universe_file), "a") as f:
            for region in self.tokenizer.universe.regions:
                f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

        # export the config (vocab size, embedding size)
        config = {
            "vocab_size": len(self.tokenizer),
            "embedding_size": self._model.embedding_dim,
        }

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)

    def encode(
        self, regions: Union[sc.AnnData, str], pooling: POOLING_TYPES = "mean"
    ) -> np.ndarray:
        """
        Get the vector for a region.

        :param Region region: Region to get the vector for.
        :param str pooling: Pooling type to use.

        :return np.ndarray: Vector for the region.
        """
        # data validation
        if not (isinstance(regions, sc.AnnData) or isinstance(regions, str)):
            raise TypeError(
                f"Regions must be of type AnnData or str, not {type(regions).__name__}"
            )
        if isinstance(regions, str):
            regions = sc.read_h5ad(regions)

        if pooling not in ["mean", "max"]:
            raise ValueError(f"pooling must be one of {POOLING_TYPES}")

        # tokenize the region
        tokens = self.tokenizer.tokenize(regions)
        tokens = [[t.id for t in sublist] for sublist in tokens]

        # get the vector
        embeddings = []
        for token_set in track(tokens, total=len(tokens), description="Getting embeddings"):
            region_embeddings = self._model.projection(torch.tensor(token_set))
            if pooling == "mean":
                region_embeddings = torch.mean(region_embeddings, axis=0).detach().numpy()
            elif pooling == "max":
                region_embeddings = torch.max(region_embeddings, axis=0).values.detach().numpy()
            else:
                # this should be unreachable
                raise ValueError(f"pooling must be one of {POOLING_TYPES}")
            embeddings.append(region_embeddings)

        return np.vstack(embeddings)
