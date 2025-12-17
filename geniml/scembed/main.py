import os
from logging import getLogger
from typing import Union

import numpy as np
import scanpy as sc
import torch
from huggingface_hub import hf_hub_download
from rich.progress import track
from gtars.tokenizers import Tokenizer

from ..region2vec.const import (
    CONFIG_FILE_NAME,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_MIN_COUNT,
    DEFAULT_WINDOW_SIZE,
    MODEL_FILE_NAME,
    POOLING_METHOD_KEY,
    POOLING_TYPES,
    UNIVERSE_FILE_NAME,
)
from ..region2vec.main import Region2Vec
from ..region2vec.utils import (
    Region2VecDataset,
    export_region2vec_model,
    load_local_region2vec_model,
    train_region2vec_model,
)
from ..tokenization.utils import tokenize_anndata
from .const import MODULE_NAME

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")


class ScEmbed:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: Tokenizer = None,
        device: str = None,
        pooling_method: POOLING_TYPES = "mean",
        **kwargs,
    ):
        """
        Initialize ScEmbed.

        Args:
            model_path (str): Path to the pre-trained model on huggingface.
            tokenizer (Tokenizer): Tokenizer instance for tokenization.
            device (str): Device to use for computation.
            pooling_method (POOLING_TYPES): Pooling method for embeddings.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        super().__init__()
        self.model_path: str = model_path
        self.tokenizer: Tokenizer
        self.trained: bool = False
        self._model: Region2Vec = None
        self.pooling_method: POOLING_TYPES = pooling_method

        if model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True

        elif tokenizer is not None:
            self._init_model(tokenizer, **kwargs)

        # set the device
        self._target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _init_tokenizer(self, tokenizer: Union[Tokenizer, str]):
        """
        Initialize the tokenizer.

        Args:
            tokenizer (Union[Tokenizer, str]): Tokenizer to add to the model.
        """
        if isinstance(tokenizer, str):
            if os.path.exists(tokenizer):
                self.tokenizer = Tokenizer(tokenizer)
            else:
                raise FileNotFoundError(f"Tokenizer file {tokenizer} not found.")
        elif isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError("tokenizer must be of type Tokenizer or str.")

    def _init_model(self, tokenizer, **kwargs):
        """
        Initialize the core model. This will initialize the model from scratch.

        Args:
            tokenizer: Tokenizer instance for the model.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        self._init_tokenizer(tokenizer)

        self._model = Region2Vec(
            len(self.tokenizer),
            embedding_dim=kwargs.get("embedding_dim", DEFAULT_EMBEDDING_DIM),
        )

    @property
    def model(self):
        """
        Get the core Region2Vec model.

        Returns:
            Region2Vec: The core Region2Vec model.
        """
        return self._model

    def add_tokenizer(self, tokenizer: Tokenizer, **kwargs):
        """
        Add a tokenizer to the model. This should be use when the model is not initialized with a tokenizer.

        Args:
            tokenizer (Tokenizer): Tokenizer to add to the model.
            **kwargs: Additional keyword arguments to pass to the model.
        """
        if self._model is not None:
            raise RuntimeError("Cannot add a tokenizer to a model that is already initialized.")

        self.tokenizer = tokenizer
        if not self.trained:
            self._init_model(**kwargs)

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.
            vocab_path (str): Path to the vocabulary file.
            config_path (str): Path to the config file.
        """
        _model, config = load_local_region2vec_model(model_path, config_path)
        tokenizer = Tokenizer(vocab_path)

        self._model = _model
        self.tokenizer = tokenizer

        if POOLING_METHOD_KEY in config:
            self.pooling_method = config[POOLING_METHOD_KEY]

    def _init_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the model from a huggingface model.

        This uses the model path to download the necessary files and then "build itself up" from those.
        This includes both the actual model and the tokenizer.

        Args:
            model_path (str): Path to the pre-trained model on huggingface.
            model_file_name (str): Name of the model file.
            universe_file_name (str): Name of the universe file.
            config_file_name (str): Name of the config file.
            **kwargs: Additional keyword arguments to pass to the hf download function.
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

        Args:
            path_to_files (str): Path to the directory containing the files.
            model_file_name (str): Name of the model file.
            universe_file_name (str): Name of the universe file.
            config_file_name (str): Name of the config file.

        Returns:
            ScEmbed: The loaded model.
        """
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(model_file_path, universe_file_path, config_file_path)
        instance.trained = True

        return instance

    def train(
        self,
        dataset: Region2VecDataset,
        window_size: int = DEFAULT_WINDOW_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        min_count: int = DEFAULT_MIN_COUNT,
        num_cpus: int = 1,
        seed: int = 42,
        save_checkpoint_path: str = None,
        gensim_params: dict = {},
        load_from_checkpoint: str = None,
    ) -> bool:
        """
        Train the model.

        Args:
            dataset (Region2VecDataset): Data to train on. This is a dataset of tokens.
            window_size (int): Window size for the model.
            epochs (int): Number of epochs to train for.
            min_count (int): Minimum count for a region to be included in the vocabulary.
            num_cpus (int): Number of cpus to use for training.
            seed (int): Seed to use for training.
            save_checkpoint_path (str): Path to save the model checkpoints to.
            gensim_params (dict): Additional parameters to pass to the gensim model.
            load_from_checkpoint (str): Path to a checkpoint to load from.

        Returns:
            bool: Whether or not the model was trained.
        """

        # validate a model exists
        if self._model is None:
            raise RuntimeError(
                "Cannot train a model that has not been initialized. Please initialize the model first using a tokenizer or from a huggingface model."
            )

        gensim_model = train_region2vec_model(
            dataset,
            embedding_dim=self._model.embedding_dim,
            window_size=window_size,
            epochs=epochs,
            min_count=min_count,
            num_cpus=num_cpus,
            seed=seed,
            save_checkpoint_path=save_checkpoint_path,
            gensim_params=gensim_params,
            load_from_checkpoint=load_from_checkpoint,
        )

        # once done training, set the weights of the pytorch model in self._model
        for id in track(
            gensim_model.wv.key_to_index,
            total=len(gensim_model.wv.key_to_index),
            description="Setting weights.",
        ):
            self._model.projection.weight.data[int(id)] = torch.tensor(gensim_model.wv[id])

        # set the model as trained
        self.trained = True

        return True

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        universe_file: str = UNIVERSE_FILE_NAME,
        config_file: str = CONFIG_FILE_NAME,
    ):
        """
        Function to facilitate exporting the model in a way that can be directly uploaded to huggingface.

        This exports the model weights and the vocabulary.

        Args:
            path (str): Path to export the model to.
            checkpoint_file (str): Name of the checkpoint file.
            universe_file (str): Name of the universe file.
            config_file (str): Name of the config file.
        """
        # make sure the model is trained
        if not self.trained:
            raise RuntimeError("Cannot export an untrained model.")

        export_region2vec_model(
            self._model,
            self.tokenizer,
            path,
            checkpoint_file=checkpoint_file,
            universe_file=universe_file,
            config_file=config_file,
        )

    def encode(self, regions: Union[sc.AnnData, str], pooling: POOLING_TYPES = None) -> np.ndarray:
        """
        Get the vector for a region.

        Args:
            regions (Union[sc.AnnData, str]): Region to get the vector for.
            pooling (POOLING_TYPES): Pooling type to use.

        Returns:
            np.ndarray: Vector for the region.
        """
        # allow the user to override the pooling method
        pooling = pooling or self.pooling_method

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
        tokens = tokenize_anndata(regions, self.tokenizer)
        tokens = [[t["input_ids"] for t in sublist] for sublist in tokens]

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
