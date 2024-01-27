from typing import Union, List

import torch
import torch.nn as nn

from ..models.main import ExModel
from ..tokenization.main import ITTokenizer
from .const import (
    POOLING_TYPES,
    POOLING_METHOD_KEY,
    DEFAULT_EMBEDDING_DIM,
    CONFIG_FILE_NAME,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
)


class Atacformer(nn.Module):
    def __init__(self, vocab_size, d_model: int = DEFAULT_EMBEDDING_DIM):
        """
        Atacformer is a transformer-based model for ATAC-seq data. It closely follows
        the architecture of BERT, but with a few modifications:
        - positional embeddings set to 0 since ATAC-seq data is not sequential
        - no next sentence prediction task
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def forward(self, x: torch.Tensor):
        """
        :param torch.Tensor x: Input tensor of shape (batch_size, seq_len)
        """
        # get the embeddings
        x = self.embedding(x)
        # set the positional embeddings to 0
        x = x + torch.zeros_like(x)
        # pass through the transformer
        x = self.transformer_encoder(x)
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
        tokenizer: ITTokenizer = None,
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
        self.tokenizer: ITTokenizer
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

    def _init_tokenizer(self, tokenizer: Union[ITTokenizer, str]):
        """
        Initialize the tokenizer.

        :param tokenizer: Tokenizer to initialize.
        """
        if isinstance(tokenizer, str):
            if os.path.exists(tokenizer):
                self.tokenizer = ITTokenizer(tokenizer)
            else:
                self.tokenizer = ITTokenizer.from_pretrained(
                    tokenizer
                )  # download from huggingface (or at least try to)
        elif isinstance(tokenizer, ITTokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError("tokenizer must be a path to a bed file or an ITTokenizer object.")

    def _init_model(self, tokenizer, **kwargs):
        """
        Initialize the core model. This will initialize the model from scratch.

        :param kwargs: Additional keyword arguments to pass to the model.
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
        """
        return self._model

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        _model, tokenizer, config = load_local_region2vec_model(
            model_path, vocab_path, config_path
        )
        self._model = _model
        self.tokenizer = tokenizer
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
    ):
        """
        Function to facilitate exporting the model in a way that can
        be directly uploaded to huggingface. This exports the model
        weights and the vocabulary.

        :param str path: Path to export the model to.
        """

        export_region2vec_model(
            self._model,
            self.tokenizer,
            path,
            checkpoint_file=checkpoint_file,
            universe_file=universe_file,
            config_file=config_file,
        )

    def encode(
        self, regions: Union[Region, List[Region]], pooling: POOLING_TYPES = None
    ) -> np.ndarray:
        """
        Get the vector for a region.

        :param Region region: Region to get the vector for.
        :param str pooling: Pooling type to use.

        :return np.ndarray: Vector for the region.
        """
        # allow for overriding the pooling method
        pooling = pooling or self.pooling_method

        # data validation
        if isinstance(regions, Region):
            regions = [regions]
        if isinstance(regions, str):
            regions = list(RegionSet(regions))
        if isinstance(regions, RegionSet):
            regions = list(regions)
        if not isinstance(regions, list):
            regions = [regions]
        if not isinstance(regions[0], Region):
            raise TypeError("regions must be a list of Region objects.")

        if pooling not in ["mean", "max"]:
            raise ValueError(f"pooling must be one of {POOLING_TYPES}")

        # tokenize the region
        tokens = [self.tokenizer.tokenize(r) for r in regions]
        tokens = [t.id for sublist in tokens for t in sublist]

        # get the vector
        region_embeddings = self._model(torch.tensor(tokens))

        if pooling == "mean":
            return torch.mean(region_embeddings, axis=0).detach().numpy()
        elif pooling == "max":
            return torch.max(region_embeddings, axis=0).values.detach().numpy()
        else:
            # this should be unreachable
            raise ValueError(f"pooling must be one of {POOLING_TYPES}")
