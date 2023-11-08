import os
from typing import Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from yaml import safe_dump, safe_load

from .const import (
    EMBEDDING_DIM_KEY,
    NUM_CLASSES_KEY,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
    CONFIG_FILE_NAME,
)
from ..region2vec.const import DEFAULT_EMBEDDING_SIZE
from ..region2vec.main import Region2Vec
from ..tokenization.main import ITTokenizer


class SingleCellTypeClassifier(nn.Module):
    def __init__(self, region2vec: Union[Region2Vec, str], num_classes: int):
        super().__init__()

        self.region2vec: Region2Vec
        if isinstance(region2vec, str):
            # is local model?
            if os.path.exists(region2vec):
                model_path = os.path.join(region2vec, MODEL_FILE_NAME)
                vocab_path = os.path.join(region2vec, UNIVERSE_FILE_NAME)
                config_path = os.path.join(region2vec, CONFIG_FILE_NAME)
                self._load_local_region2vec_model(model_path, vocab_path, config_path)
            else:
                self._init_region2vec_from_huggingface(region2vec)
        elif isinstance(region2vec, Region2Vec):
            self.region2vec = region2vec
        else:
            raise ValueError(
                "Region2vec must be either a hugginface registry path, a path to a local model or an instance of Region2Vec."
            )

        self.embedding_dim = self.region2vec.embedding_dim
        self.num_classes = num_classes
        self.output_layer = nn.Linear(self.region2vec.embedding_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.region2vec(x)
        x = x.sum(dim=1)
        x = self.output_layer(x)
        return x

    def _load_local_region2vec_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load a Region2Vec model from a checkpoint.

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

        self.region2vec = Region2Vec(
            config["vocab_size"],
            embedding_dim=config["embedding_size"],
        )
        self.region2vec.load_state_dict(params)

    def _init_region2vec_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the Region2Vec model from a huggingface model. This uses the model path
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

        self._load_local_region2vec_model(model_file_path, universe_path, config_path)


class SingleCellTypeClassifierExModel:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: ITTokenizer = None,
        device: str = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.device = device

        if model_path is not None:
            self.__init_from_huggingface(model_path)
        else:
            self.__init_model(**kwargs)

    def _init_from_huggingface(self, model_path: str):
        raise NotImplementedError("Not implemented yet.")

    def _init_model(self, **kwargs):
        if self.tokenizer is None:
            raise ValueError("A tokenizer must be provided when creating a model from scratch.")
        if kwargs.get(NUM_CLASSES_KEY) is None:
            raise ValueError(
                "Number of classes must be provided when creating a model from scratch."
            )

        embedding_dim = kwargs.get(EMBEDDING_DIM_KEY, DEFAULT_EMBEDDING_SIZE)
        num_classes = kwargs.get(NUM_CLASSES_KEY)
        r2v = Region2Vec(
            len(self.tokenizer),
            embedding_dim,
        )
        self._vocab_len = len(self.tokenizer)
        self._model = SingleCellTypeClassifier(r2v, num_classes)

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
            "num_classes": self._model.num_classes,
        }

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)
