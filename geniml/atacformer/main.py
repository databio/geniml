import os
import math
from typing import Union

import torch
import torch.nn as nn
import tomllib
import scanpy as sc
from huggingface_hub import hf_hub_download
from yaml import safe_dump, safe_load

from ..models.main import ExModel
from ..tokenization.main import AnnDataTokenizer
from .const import (
    CONFIG_FILE_NAME,
    D_MODEL_KEY,
    MODEL_FILE_NAME,
    N_HEADS_KEY,
    N_LAYERS_KEY,
    POOLING_METHOD_KEY,
    POOLING_TYPES,
    UNIVERSE_CONFIG_FILE_NAME,
    UNIVERSE_FILE_NAME,
    VOCAB_SIZE_KEY,
    CONTEXT_SIZE_KEY,
    D_FF_KEY,
)


class Atacformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_position_embeddings: int = 2048,
        positional_encoding: str = "sinusoidal",
    ):
        """
        Atacformer is a transformer-based model for ATAC-seq data. It closely follows
        the architecture of BERT, but with a few modifications:
        - positional embeddings set to 0 since ATAC-seq data is not sequential
        - no next sentence prediction task
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.vocab_size = vocab_size

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # positional encoding
        self.positional_encoding_type = positional_encoding
        if positional_encoding == "sinusoidal":
            self.positional_encoding = self._create_sinusoidal_positional_encoding(
                max_position_embeddings, d_model
            )
        elif positional_encoding == "learned":
            self.positional_encoding = nn.Embedding(max_position_embeddings, d_model)
        else:
            raise ValueError("Invalid positional encoding type. Choose 'sinusoidal' or 'learned'.")

        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=d_ff,
            norm_first=True,
        )

        # stack the encoder layers
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

    @staticmethod
    def _create_sinusoidal_positional_encoding(max_len, d_model):
        """
        Create a sinusoidal positional encoding matrix.
        :param max_len: Maximum sequence length.
        :param d_model: Embedding dimension.
        :return: Tensor of shape (max_len, d_model)
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension for broadcasting
        return pe

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape (batch_size, seq_len)
        :param torch.Tensor mask: Mask tensor of shape (batch_size, seq_len)
        :return torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model). I.e. an embedding for each token.
        """
        # get the embeddings
        x = self.embedding(x)

        # set the positional embeddings to 0
        x = x + torch.zeros_like(x)

        # pass through the transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        return x


class AtacformerExModel(ExModel):
    """
    An "extended model" for Atacformer. This class provides a wrapper around the
    Atacformer model, and provides additional functionality such as tokenization
    and encoding of new data.
    """

    def __init__(
        self,
        model_path: str = None,
        tokenizer: AnnDataTokenizer = None,
        pooling_method: POOLING_TYPES = "mean",
        device: str = None,
        context_size: int = 2048,
        **kwargs,
    ):
        """
        Initialize Region2VecExModel.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param embedding_dim: Dimension of the embedding.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        super().__init__()
        self.model_path: str = model_path
        self.tokenizer: AnnDataTokenizer
        self.trained: bool = False
        self._model: Atacformer = None
        self.pooling_method = pooling_method
        self.context_size = context_size

        if model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True

        elif tokenizer is not None:
            self._init_model(tokenizer, **kwargs)

        # set the device
        self._target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if self._model is not None:
            self._model = self._model.to(self._target_device)

    def _init_tokenizer(self, tokenizer: Union[AnnDataTokenizer, str]):
        """
        Initialize the tokenizer.

        :param tokenizer: Tokenizer to initialize.
        """
        if isinstance(tokenizer, str):
            if os.path.exists(tokenizer):
                self.tokenizer = AnnDataTokenizer(tokenizer)
            else:
                self.tokenizer = AnnDataTokenizer.from_pretrained(
                    tokenizer
                )  # download from huggingface (or at least try to)
        elif isinstance(tokenizer, AnnDataTokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "tokenizer must be a path to a bed file or an `AnnDataTokenizer object."
            )

    def _init_model(self, tokenizer, **kwargs):
        """
        Initialize the core model. This will initialize the model from scratch.

        :param kwargs: Additional keyword arguments to pass to the model.
        """
        self._init_tokenizer(tokenizer)
        self._model = Atacformer(
            len(self.tokenizer),
            d_model=kwargs.get("d_model", 768),
            n_heads=kwargs.get("n_heads", 12),
            n_layers=kwargs.get("n_layers", 12),
            d_ff=kwargs.get("d_ff", 3072),
        )

    @property
    def model(self):
        """
        Get the core Region2Vec model.
        """
        return self._model

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """

        # check the type of the tokenizer
        # open the toml file and read the `tokenizer_type` key
        with open(vocab_path, "rb") as fp:
            tokenizer_config = tomllib.load(fp)

        tokenizer_type = tokenizer_config.get("tokenizer_type", "tree")

        # init the tokenizer - only one option for now
        self.tokenizer = AnnDataTokenizer(vocab_path, tokenizer_type=tokenizer_type)

        # load the model state dict (weights)
        params = torch.load(model_path)

        # get the model config (vocab size, embedding size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        # try with new key first, then old key for backwards compatibility
        d_model = config.get(D_MODEL_KEY, None)
        if d_model is None:
            raise KeyError(
                f"Could not find embedding dimension in config file. Expected key {D_MODEL_KEY}."
            )

        n_heads = config.get(N_HEADS_KEY, None)
        if n_heads is None:
            raise KeyError(f"Could not find n_heads in config file. Expected key {N_HEADS_KEY}.")

        n_layers = config.get(N_LAYERS_KEY, None)
        if n_layers is None:
            raise KeyError(f"Could not find n_layers in config file. Expected key {N_LAYERS_KEY}.")

        d_ff = config.get(D_FF_KEY, None)
        if d_ff is None:
            raise KeyError(f"Could not find d_ff in config file. Expected key {D_FF_KEY}.")

        vocab_size = config.get(VOCAB_SIZE_KEY) or len(self.tokenizer)

        model = Atacformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )
        model.load_state_dict(params)

        self._model = model

        self.trained = True
        if POOLING_METHOD_KEY in config:
            self.pooling_method = config[POOLING_METHOD_KEY]

    def _init_from_huggingface(
        self,
        model_path: str,
        **kwargs,
    ):
        """
        Initialize the model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those. This
        includes both the actual model and the tokenizer.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param str model_file_name: Name of the model file.
        :param str universe_config_file_name: Name of the universe file.
        :param kwargs: Additional keyword arguments to pass to the hf download function.
        """
        model_file_name: str = MODEL_FILE_NAME
        universe_config_file_name: str = UNIVERSE_CONFIG_FILE_NAME
        universe_file_name: str = UNIVERSE_FILE_NAME
        config_file_name: str = CONFIG_FILE_NAME

        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_config_path = hf_hub_download(model_path, universe_config_file_name, **kwargs)
        _universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self._load_local_model(model_file_path, universe_config_path, config_path)

    @classmethod
    def from_pretrained(cls, path_to_files: str) -> "AtacformerExModel":
        """
        Load the model from a set of files that were exported using the export function.

        :param str path_to_files: Path to the directory containing the files.
        :param str model_file_name: Name of the model file.
        :param str universe_config_file_name: Name of the universe file.
        """
        model_file_name: str = MODEL_FILE_NAME
        universe_config_file_name: str = UNIVERSE_CONFIG_FILE_NAME
        config_file_name: str = CONFIG_FILE_NAME

        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_config_file_path = os.path.join(path_to_files, universe_config_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(model_file_path, universe_config_file_path, config_file_path)
        instance.trained = True

        return instance

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        _universe_config_file: str = UNIVERSE_CONFIG_FILE_NAME,
        config_file: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Function to facilitate exporting the model in a way that can
        be directly uploaded to huggingface. This exports the model
        weights and the vocabulary.

        :param str path: Path to export the model to.
        """
        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # make sure the model is unwrapped
        if isinstance(self._model, nn.DataParallel):
            self._model = self._model.module
        elif isinstance(self._model, nn.parallel.DistributedDataParallel):
            self._model = self._model.module
        elif isinstance(self._model, nn.Module):
            pass
        else:
            raise TypeError("model must be an nn.Module object.")

        # export the model weights
        torch.save(self._model.state_dict(), os.path.join(path, checkpoint_file))

        d_model = self._model.d_model
        n_layers = self._model.n_layers
        n_heads = self._model.n_heads
        d_ff = self._model.d_ff
        vocab_size = len(self.tokenizer)
        context_size = self.context_size

        config = {
            POOLING_METHOD_KEY: self.pooling_method,
            D_MODEL_KEY: d_model,
            VOCAB_SIZE_KEY: vocab_size,
            N_HEADS_KEY: n_heads,
            N_LAYERS_KEY: n_layers,
            D_FF_KEY: d_ff,
            CONTEXT_SIZE_KEY: context_size,
        }

        if kwargs:
            config.update(kwargs)

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)

    def encode(self, adata: Union[str, sc.AnnData]) -> torch.Tensor:
        """
        Get the vector for a region.

        :param Region region: Region to get the vector for.
        :param str pooling: Pooling type to use.

        :return np.ndarray: Vector for the region.
        """
        raise NotImplementedError("This method is not implemented yet. Stay tuned...")
