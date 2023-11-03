import logging
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec as GensimWord2Vec
from huggingface_hub import hf_hub_download
from rich.progress import track
from yaml import safe_load, safe_dump

from ..tokenization.main import ITTokenizer, Tokenizer
from ..io.io import RegionSet, Region
from ..const import PKG_NAME
from .const import (
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_COUNT,
    CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
)
from .utils import shuffle_documents, LearningRateScheduler

_LOGGER = logging.getLogger(PKG_NAME)


class Word2Vec(nn.Module):
    """
    Word2Vec model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.projection = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        return x


class Region2Vec(Word2Vec):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
    ):
        super().__init__(vocab_size, embedding_dim)


class Region2VecExModel:
    def __init__(
        self, model_path: str = None, tokenizer: ITTokenizer = None, device: str = None, **kwargs
    ):
        """
        Initialize Region2VecExModel.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param embedding_dim: Dimension of the embedding.
        :param hidden_dim: Dimension of the hidden layer.
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
            self._model = Word2Vec(
                len(self.tokenizer),
                embedding_dim=kwargs.get("embedding_dim", DEFAULT_EMBEDDING_SIZE),
            )

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

        # get the model config (vocab size, embedding size, hidden size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self._model = Region2Vec(
            config["vocab_size"],
            embedding_dim=config["embedding_size"],
            hidden_dim=config["hidden_size"],
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
    ) -> "Region2VecExModel":
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

    def _validate_data_for_training(
        self, data: Union[List[RegionSet], List[str], List[List[Region]]]
    ) -> List[RegionSet]:
        """
        Validate the data for training. This will return a list of RegionSets if the data is valid.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                       a list of RegionSets or a list of paths to bed files.
        :return: List of RegionSets.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list or RegionSets or a list of paths to bed files.")
        if len(data) == 0:
            raise ValueError("data must not be empty.")

        # check if the data is a list of RegionSets
        if isinstance(data[0], RegionSet):
            return data
        elif isinstance(data[0], str):
            return [RegionSet(f) for f in data]
        elif isinstance(data[0], list) and isinstance(data[0][0], Region):
            return [RegionSet([r for r in region_list]) for region_list in data]

    def train(
        self,
        data: Union[List[RegionSet], List[str]],
        window_size: int = DEFAULT_WINDOW_SIZE,
        epochs: int = DEFAULT_EPOCHS,
        min_count: int = DEFAULT_MIN_COUNT,
        num_cpus: int = 1,
        seed: int = 42,
        checkpoint_path: str = MODEL_FILE_NAME,
        save_model: bool = False,
        gensim_params: dict = {},
    ) -> np.ndarray:
        """
        Train the model.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                        a list of RegionSets or a list of paths to bed files.
        :param int window_size: Window size for the model.
        :param int epochs: Number of epochs to train for.
        :param int min_count: Minimum count for a region to be included in the vocabulary.
        :param int n_shuffles: Number of shuffles to perform on the data.
        :param int batch_size: Batch size for training.
        :param str checkpoint_path: Path to save the model checkpoint to.
        :param torch.optim.Optimizer optimizer: Optimizer to use for training.
        :param float learning_rate: Learning rate to use for training.
        :param int ns_k: Number of negative samples to use.
        :param torch.device device: Device to use for training.
        :param dict optimizer_params: Additional parameters to pass to the optimizer.
        :param bool save_model: Whether or not to save the model.

        :return np.ndarray: Loss values for each epoch.
        """
        # validate a model exists
        if self._model is None:
            raise RuntimeError(
                "Cannot train a model that has not been initialized. Please initialize the model first using a tokenizer or from a huggingface model."
            )

        # validate the data - convert all to RegionSets
        _LOGGER.info("Validating data for training.")
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

        # tokenize the data
        # convert to strings for gensim
        _LOGGER.info("Tokenizing data.")
        tokenized_data = [self.tokenizer.tokenize(region_set) for region_set in data]
        tokenized_data = [[str(t.id) for t in region_set] for region_set in tokenized_data]
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
        for id in gensim_model.wv.key_to_index:
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
        for region in self.tokenizer.universe.regions:
            with open(os.path.join(path, universe_file), "a") as f:
                f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

        # export the config (vocab size, embedding size)
        config = {
            "vocab_size": len(self.tokenizer),
            "embedding_size": self._model.embedding_dim,
            "hidden_size": self._model.hidden.out_features,
        }

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)

    def encode(self, region: Region, pooling: str = "mean") -> np.ndarray:
        """
        Get the vector for a region.

        :param Region region: Region to get the vector for.
        :return np.ndarray: Vector for the region.
        """
        if not self.trained:
            raise RuntimeError("Cannot get a vector from an untrained model.")

        if pooling != "mean":
            raise NotImplementedError("Only mean pooling is currently supported.")

        # tokenize the region
        tokens = self.tokenizer.tokenize(region)
        tokens = [t.id for t in tokens]

        # get the vector
        region_embeddings = self._model.projection(torch.tensor(tokens))

        return torch.mean(region_embeddings, dim=0).detach().numpy()
