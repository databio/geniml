import os
from logging import getLogger
from typing import Union

import numpy as np
import scanpy as sc
import torch
from huggingface_hub import hf_hub_download
from rich.progress import track
from yaml import safe_load

from ..models.main import ExModel
from ..region2vec.utils import LearningRateScheduler, shuffle_documents
from ..region2vec.experimental import Word2Vec
from ..tokenization import InMemTokenizer, Tokenizer
from .const import (
    CHR_KEY,
    END_KEY,
    MODEL_FILE_NAME,
    MODULE_NAME,
    START_KEY,
    UNIVERSE_FILE_NAME,
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_COUNT,
    CONFIG_FILE_NAME,
)

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")


class ScEmbed(ExModel):
    """
    ScEmbed extended model for single-cell ATAC-seq data. It is a single-cell
    extension of Region2Vec.
    """

    def __init__(
        self,
        model_path: str = None,
        tokenizer: InMemTokenizer = None,
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
        self.tokenizer: InMemTokenizer = tokenizer
        self.trained: bool = False
        self._model: Word2Vec = None

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
            self._model = Word2Vec(
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
        self.tokenizer = InMemTokenizer(vocab_path)

        # load the model state dict (weights)
        params = torch.load(model_path)

        # get the model config (vocab size, embedding size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self._model = Word2Vec(
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

    def _validate_data(self, data: Union[sc.AnnData, str]) -> sc.AnnData:
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
        :param kwargs: Keyword arguments to pass to the model training function.
        """
        from gensim.models import Word2Vec as GensimWord2Vec

        _LOGGER.info("Validating data.")
        data = self._validate_data(data)

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
                tokenized_data, total=len(tokenized_data), description="Converting to strings."
            )
        ]
        gensim_model.build_vocab(tokenized_data)

        # create our own learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            n_epochs=epochs,
        )

        # train the model
        losses = []

        for epoch in track(range(epochs), description="Training model"):
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

    def export(self, path: str):
        """
        Export a model for direct upload to the HuggingFace Hub.

        :param str path: The path to save the model to.
        """
        # make folder path if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        model_file_path = os.path.join(path, MODEL_FILE_NAME)
        universe_file_path = os.path.join(path, UNIVERSE_FILE_NAME)

        # save the model
        self._model.save(model_file_path)

        # save universe (vocab)
        with open(universe_file_path, "w") as f:
            for region in self.tokenizer.universe:
                f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

    def upload_to_huggingface(self, model_name: str, token: str = None, **kwargs):
        """
        Upload the model to the HuggingFace Hub.

        :param str model_name: The name of the model to upload.
        :param kwargs: Additional keyword arguments to pass to the upload function.
        """
        raise NotImplementedError("This method is not yet implemented.")

    def encode(self, adata: sc.AnnData) -> np.ndarray:
        """
        Encode the data into a latent space.

        :param sc.AnnData adata: The AnnData object containing the data to encode.
        :return np.ndarray: The encoded data.
        """
        # tokenize the data
        region_sets = self.tokenizer.tokenize(adata)

        # encode the data
        _LOGGER.info("Encoding data.")
        enoded_data = []
        for region_set in track(region_sets, description="Encoding data", total=len(region_sets)):
            vectors = self._model.forward(region_set)
            # compute the mean of the vectors
            vector = np.mean(vectors, axis=0)
            enoded_data.append(vector)
        return np.array(enoded_data)

    def __call__(self, adata: sc.AnnData) -> np.ndarray:
        return self.encode(adata)
