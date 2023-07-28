import os
from logging import getLogger
from typing import List, Union

import numpy as np
import scanpy as sc
from huggingface_hub import hf_hub_download, upload_file, login
from numba import config
from tqdm import tqdm

from ..io import Region, RegionSet
from ..models.main import ExModel
from ..region2vec import Region2Vec
from ..tokenization import InMemTokenizer
from .const import CHR_KEY, END_KEY, MODEL_FILE_NAME, MODULE_NAME, START_KEY, UNIVERSE_FILE_NAME
from .utils import make_syn1neg_file_name, make_wv_file_name

_GENSIM_LOGGER = getLogger("gensim")
_LOGGER = getLogger(MODULE_NAME)

# demote gensim logger to warning
_GENSIM_LOGGER.setLevel("WARNING")

# set the threading layer before any parallel target compilation
config.THREADING_LAYER = "threadsafe"  # type: ignore


class ScEmbed(ExModel):
    """
    ScEmbed model for single-cell ATAC-seq data. It is a single-cell
    extension of Region2Vec.
    """

    def __init__(self, model_repo: Union[str, None] = None, **kwargs):
        """
        Initialize ScEmbed model.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param kwargs: Additional keyword arguments to pass to the model.
        """
        self.model_path = model_repo
        self._model: Region2Vec = None
        self.tokenizer: InMemTokenizer = None

        if model_repo is not None:
            self._init_from_huggingface(model_repo)
        else:
            self._model = Region2Vec(**kwargs)
            self.tokenizer = InMemTokenizer()

    @property
    def wv(self):
        return self._model.wv

    @property
    def trained(self):
        return self._model.trained

    def from_pretrained(self, model_file_path: str, universe_file_path: str):
        """
        Initialize ScEmbed model from pretrained model.

        :param str model_file_path: Path to the pre-trained model.
        :param str universe_file_path: Path to the universe file.
        """
        self._model = Region2Vec.load(model_file_path)
        self.tokenizer = InMemTokenizer(universe_file_path)

    def _init_from_huggingface(
        self,
        model_repo: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        **kwargs,
    ):
        """
        Download a pretrained model from the HuggingFace Hub. We need to download
        the actual model + weights, and the universe file.

        :param str model: The name of the model to download (this is the same as the repo name).
        :param str model_file_name: The name of the model file - this should almost never be changed.
        :param str universe_file_name: The name of the universe file - this should almost never be changed.
        """
        model_path = hf_hub_download(model_repo, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_repo, universe_file_name, **kwargs)

        syn1reg_file_name = make_syn1neg_file_name(model_file_name)
        wv_file_name = make_wv_file_name(model_file_name)

        # get the syn1neg and wv files
        hf_hub_download(model_repo, wv_file_name, **kwargs)
        hf_hub_download(model_repo, syn1reg_file_name, **kwargs)

        # set the paths to the downloaded files
        self._model_path = model_path
        self._universe_path = universe_path

        # load the model
        self._model = Region2Vec.load(model_path)
        self.tokenizer = InMemTokenizer(universe_path)

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

    def train(self, data: Union[sc.AnnData, str], **kwargs):
        """
        Train the model.

        :param sc.AnnData data: The AnnData object containing the data to train on (can be path to AnnData).
        :param kwargs: Keyword arguments to pass to the model training function.
        """
        _LOGGER.info("Validating data.")
        data = self._validate_data(data)

        # extract out the chr, start, end columns
        chrs = data.var[CHR_KEY].values.tolist()
        starts = data.var[START_KEY].values.tolist()
        ends = data.var[END_KEY].values.tolist()
        regions = [Region(c, int(s), int(e)) for c, s, e in zip(chrs, starts, ends)]

        _LOGGER.info("Extracting region sets.")

        # fit the tokenizer on the regions
        self.tokenizer.fit(regions)

        # convert the data to a list of documents
        region_sets = self.tokenizer.tokenize(data)

        _LOGGER.info("Training begin.")
        self._model.train(region_sets, **kwargs)

        _LOGGER.info("Training complete.")

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
        for region_set in tqdm(region_sets, desc="Encoding data", total=len(region_sets)):
            vectors = self._model.forward(region_set)
            # compute the mean of the vectors
            vector = np.mean(vectors, axis=0)
            enoded_data.append(vector)
        return np.array(enoded_data)

    def __call__(self, adata: sc.AnnData) -> np.ndarray:
        return self.encode(adata)
