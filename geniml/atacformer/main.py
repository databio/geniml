import os
from typing import Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from yaml import safe_dump, safe_load

from ..models.main import ExModel
from ..tokenization.main import AnnDataTokenizer
from .const import (
    CONFIG_FILE_NAME,
    D_MODEL_KEY,
    DEFAULT_EMBEDDING_DIM,
    MODEL_FILE_NAME,
    NHEAD_KEY,
    NUM_LAYERS_KEY,
    POOLING_METHOD_KEY,
    POOLING_TYPES,
    UNIVERSE_FILE_NAME,
    VOCAB_SIZE_KEY,
)


class Atacformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = DEFAULT_EMBEDDING_DIM,
        nhead: int = 8,
        num_layers: int = 6,
        context_size: int = 2048,
    ):
        """
        Atacformer is a transformer-based model for ATAC-seq data. It closely follows
        the architecture of BERT, but with a few modifications:
        - positional embeddings set to 0 since ATAC-seq data is not sequential
        - no next sentence prediction task
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        :param torch.Tensor x: Input tensor of shape (batch_size, seq_len)
        :param torch.Tensor mask: Mask tensor of shape (batch_size, seq_len)
        :return torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model). I.e. an embedding for each token.
        """
        # get the embeddings
        x = self.embedding(x)
        # set the positional embeddings to 0
        # x = x + torch.zeros_like(x)

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
        device: str = None,
        pooling_method: POOLING_TYPES = "mean",
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

        if model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True

        elif tokenizer is not None:
            self._init_model(tokenizer, **kwargs)

        # set the device
        self._target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

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
            d_model=kwargs.get("d_model", DEFAULT_EMBEDDING_DIM),
            nhead=kwargs.get("nhead", 8),
            num_layers=kwargs.get("num_layers", 6),
            context_size=kwargs.get("context_size", 2048),
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
        # init the tokenizer - only one option for now
        self.tokenizer = AnnDataTokenizer(vocab_path)

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

        nhead = config.get(NHEAD_KEY, None)
        if nhead is None:
            raise KeyError(f"Could not find nhead in config file. Expected key {NHEAD_KEY}.")

        num_layers = config.get(NUM_LAYERS_KEY, None)
        if num_layers is None:
            raise KeyError(
                f"Could not find num_layers in config file. Expected key {NUM_LAYERS_KEY}."
            )

        vocab_size = config.get(VOCAB_SIZE_KEY) or len(self.tokenizer)

        model = Atacformer(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers
        )
        model.load_state_dict(params)

        self.trained = True
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
    ) -> "AtacformerExModel":
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

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        universe_file: str = UNIVERSE_FILE_NAME,
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

        # export the vocabulary
        with open(os.path.join(path, universe_file), "a") as f:
            for region in self.tokenizer.universe.regions:
                f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

        d_model = self._model.d_model
        num_layers = self._model.num_layers
        nhead = self._model.nhead
        vocab_size = len(self.tokenizer)

        config = {
            POOLING_METHOD_KEY: self.pooling_method,
            D_MODEL_KEY: d_model,
            VOCAB_SIZE_KEY: vocab_size,
            NUM_LAYERS_KEY: num_layers,
            NHEAD_KEY: nhead,
        }

        if kwargs:
            config.update(kwargs)

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)

    def encode(self):
        """
        Get the vector for a region.

        :param Region region: Region to get the vector for.
        :param str pooling: Pooling type to use.

        :return np.ndarray: Vector for the region.
        """
        # TODO: write this function
