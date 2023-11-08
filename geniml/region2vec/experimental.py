import logging
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from rich.progress import track
from rich.progress import Progress
from torch.utils.data import DataLoader
from yaml import safe_load, safe_dump

from ..tokenization.main import ITTokenizer, Tokenizer
from ..io.io import RegionSet, Region
from ..const import PKG_NAME
from .const import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_COUNT,
    DEFAULT_N_SHUFFLES,
    DEFAULT_OPTIMIZER,
    DEFAULT_INIT_LR,
    CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
)
from .utils import (
    generate_window_training_data,
    Region2VecDataset,
    NegativeSampler,
    NSLoss,
    generate_frequency_distribution,
)

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
        n_shuffles: int = DEFAULT_N_SHUFFLES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        checkpoint_path: str = MODEL_FILE_NAME,
        optimizer: torch.optim.Optimizer = DEFAULT_OPTIMIZER,
        learning_rate: float = DEFAULT_INIT_LR,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: Union[str, List[str]] = "cpu",
        optimizer_params: dict = {},
        save_model: bool = False,
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
        :param torch.nn.modules.loss._Loss loss_fn: Loss function to use for training.
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

        # gtokenize the data into universe regions (recognized by this model's vocabulary)
        _LOGGER.info("Tokenizing data.")
        tokens = [
            self.tokenizer.tokenize(list(rs))
            for rs in track(data, total=len(data), description="Tokenizing")
            if len(rs) > 0  # ignore empty region sets
        ]
        tokens = [[t.id for t in tokens_list] for tokens_list in tokens]

        # generate frequency distribution
        _LOGGER.info("Generating frequency distribution.")
        freq_dist = generate_frequency_distribution(tokens, len(self._model.vocab_size))

        # create the dataset of windows
        _LOGGER.info("Generating contexts and targets.")
        _padding_token = self.tokenizer.padding_token()
        samples = generate_window_training_data(
            tokens, window_size, n_shuffles, min_count, padding_value=_padding_token.id
        )
        dataset = Region2VecDataset(samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        nsampler = NegativeSampler(freq_dist)

        # init the optimizer
        optimizer = optimizer(
            self._model.parameters(),
            lr=learning_rate,
            **optimizer_params,
        )

        # init scheduler if passed
        if learning_rate_scheduler is not None:
            learning_rate_scheduler = learning_rate_scheduler(optimizer)

        # init the loss function
        loss_fn = NSLoss()

        # move necessary things to the device
        if isinstance(device, list):
            # _LOGGER.info(f"Training on {len(device)} devices.")
            self._model = nn.DataParallel(self._model)
            self._model.to(device[0])
        else:
            self._model.to(device)

        # losses
        losses = []

        # this is ok for parallelism because each GPU will have its own copy of the model
        if isinstance(device, list):
            tensor_device = device[0]
        else:
            tensor_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # train the model for the specified number of epochs
        _LOGGER.info("Training begin.")
        with Progress() as progress_bar:
            epoch_tid = progress_bar.add_task("Epochs", total=epochs)
            batches_tid = progress_bar.add_task("Batches", total=len(dataloader))
            for epoch in range(epochs):
                for i, batch in enumerate(dataloader):
                    # zero the gradients
                    optimizer.zero_grad()
                    # get the context and target
                    context, target = batch
                    # move to device
                    context = context.to(tensor_device)
                    target = target.to(tensor_device)

                    # forward pass
                    pred = self._model(context)

                    # backward pass - SoftMax is included in the loss function
                    loss = loss_fn(pred, target)
                    loss.backward()

                    # update parameters
                    optimizer.step()

                    # update learning rate if necessary
                    if learning_rate_scheduler is not None:
                        learning_rate_scheduler.step()

                    # update progress bar
                    progress_bar.update(batches_tid, completed=i + 1)

                # update progress bar
                progress_bar.update(epoch_tid, completed=epoch + 1)

                # log out loss
                _LOGGER.info(f"Epoch {epoch + 1} loss: {loss.item()}")
                losses.append(loss.item())

        # save the model
        self.trained = True

        if save_model:
            torch.save(self._model.state_dict(), checkpoint_path)

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
