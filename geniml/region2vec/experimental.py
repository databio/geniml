import logging
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from rich.progress import track
from torch.utils.data import DataLoader

from ..tokenization.main import ITTokenizer, Tokenizer
from ..models.main import ExModel
from ..io.io import RegionSet, Region
from ..const import PKG_NAME
from .const import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_EPOCHS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_COUNT,
    DEFAULT_N_SHUFFLES,
    DEFAULT_OPTIMIZER,
    DEFAULT_LOSS_FN,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
)
from .utils import generate_window_training_data, Region2VecDataset

_LOGGER = logging.getLogger(PKG_NAME)


class Word2Vec(nn.Module):
    """
    Word2Vec model. This is the CBOW model.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.projection = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = torch.sum(x, dim=0)
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return F.log_softmax(x, dim=0)


class Region2Vec(Word2Vec):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = DEFAULT_EMBEDDING_SIZE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ):
        super().__init__(vocab_size, embedding_dim, hidden_dim)


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
                hidden_dim=kwargs.get("hidden_dim", DEFAULT_HIDDEN_DIM),
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

    def _load_local_model(self, model_path: str, vocab_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        self._model_path = model_path
        self._universe_path = vocab_path

        self.tokenizer = ITTokenizer(vocab_path)

        params = torch.load(model_path)
        self._model = Region2Vec.load_state_dict(params)

    def _init_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
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
        model_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)

        self._load_local_model(model_path, universe_path)

    def load(
        self,
        path_to_files: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
    ):
        """
        Load the model from a set of files that were exported using the export function.

        :param str path_to_files: Path to the directory containing the files.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        """
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)

        self._load_local_model(model_file_path, universe_file_path)

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
        n_shuffles: int = DEFAULT_N_SHUFFLES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        checkpoint_path: str = MODEL_FILE_NAME,
        optimizer: torch.optim.Optimizer = DEFAULT_OPTIMIZER,
        loss_fn: torch.nn.modules.loss._Loss = DEFAULT_LOSS_FN,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """
        Train the model.

        :param Union[List[RegionSet], List[str]] data: List of data to train on. This is either
                                                        a list of RegionSets or a list of paths to bed files.
        :param int window_size: Window size for the model.
        :param int epochs: Number of epochs to train for.
        :param int min_count: Minimum count for a region to be included in the vocabulary.
        :param int n_shuffles: Number of shuffles to perform on the data.

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

        # tokenize the data into regions
        _LOGGER.info("Tokenizing data.")
        tokens = [self.tokenizer.tokenize(list(rs)) for rs in track(data, total=len(data))]
        tokens = [[t.id for t in tokens_list] for tokens_list in tokens]

        _padding_token = self.tokenizer.padding_token()

        # create the dataset of windows
        _LOGGER.info("Generating contexts and targets.")
        contexts, targets = generate_window_training_data(
            tokens, window_size, n_shuffles, min_count, padding_value=_padding_token.id
        )
        dataset = Region2VecDataset(contexts, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # init
        optimizer = optimizer(self._model.parameters())

        # move necessary things to the device
        self._model.to(device)

        # losses
        losses = []

        # train the model for the specified number of epochs
        _LOGGER.info("Training begin.")
        for epoch in track(range(epochs), desc="Epochs"):
            for batch in track(iter(dataloader), desc="Batches"):
                # zero the gradients
                optimizer.zero_grad()

                context, target = batch

                # convert target to one-hot
                # this is necessary for CrossEntropyLoss
                y = torch.nn.functional.one_hot(target, self._vocab_length)
                y = y.type(torch.float)

                # forward pass
                y_pred = self._model(torch.stack(context))

                # backward pass
                loss = loss_fn(y_pred, target)
                loss.backward()

                # update parameters
                optimizer.step()

            # log out loss
            _LOGGER.info(f"Epoch {epoch + 1} loss: {loss.item()}")
            losses.append(loss.item())

        # save the model
        self.trained = True
        torch.save(self._model.state_dict(), checkpoint_path)

        return np.array(losses)

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        universe_file: str = UNIVERSE_FILE_NAME,
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
        torch.save(self._model.state_dict(), checkpoint_file)

        # export the vocabulary
        for regionin in self.tokenizer.universe:
            with open(os.path.join(path, universe_file), "a") as f:
                f.write(regionin + "\n")
