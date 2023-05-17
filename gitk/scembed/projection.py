import logging
import os
from typing import Dict, List

import numpy as np
import scanpy as sc
import yacman
from tqdm import tqdm

from .const import MODEL_CACHE_DIR, MODEL_HUB_URL, MODULE_NAME, MODEL_CONFIG_FILE_NAME
from .utils import (
    load_scembed_model,
    check_model_exists_on_hub,
    download_remote_model,
    load_universe_file,
    generate_var_conversion_map,
)
from .models import ModelCard, Universe

_LOGGER = logging.getLogger(MODULE_NAME)


class Projector:
    def __init__(self, path_to_model: str, no_cache: bool = False):
        """
        Initialize a projector object.

        :param str path_to_model: The path to the model. This can be a local path or a registry name ont he model hub.
        :param bool no_cache: If True, will not use the cache directory.
        """
        self.path_to_model = path_to_model
        self.model_config: ModelCard = None
        self.model: Dict[str, np.ndarray] = None
        self.universe: Universe = None
        self.no_cache = no_cache
        self._load_model()

    def _make_cache_dir(self):
        """
        Make the cache directory if it doesn't exist.
        """
        _LOGGER.debug(f"Making cache directory: {MODEL_CACHE_DIR}")
        if not os.path.exists(MODEL_CACHE_DIR):
            os.mkdir(MODEL_CACHE_DIR)

    def _load_local_model(self, path: str):
        """
        Will load in a local model from the path_to_model attribute. It
        is assumed that this is a pickel-dumped scembed.SCEmbed() model using
        the model.save_model() method.

        The `path` parameter should be a path to a config `yaml` file that
        contains the path to the model.

        :param str path: The path to the model.
        """
        _LOGGER.debug(f"Loading local model from {path}")
        # read yaml file
        yacmap = yacman.YacAttMap(filepath=path)
        self.model_config = ModelCard(**yacmap.to_dict()["model"])

        # load in the scEmbed model, build path to weights. It sits
        # next to the config file.
        _model = load_scembed_model(
            os.path.join(os.path.dirname(path), self.model_config.path_to_weights)
        )

        # load in the universe
        path_to_universe = os.path.join(
            os.path.dirname(path), self.model_config.path_to_universe
        )
        universe_regions = load_universe_file(path_to_universe)
        self.universe = Universe(
            reference=self.model_config.reference, regions=universe_regions
        )

        # we only want the region2vec dictionary from this
        self.model = _model.region2vec

    def _load_remote_model(self, registry: str):
        """
        Load a model from the model-hub using the reigstry path.
        """
        # look for model in cache, should be at cache/registry
        local_model_path = os.path.join(
            MODEL_CACHE_DIR, registry, MODEL_CONFIG_FILE_NAME
        )

        # if exists and no_cache is False, load it
        if os.path.exists(local_model_path) and not self.no_cache:
            self._load_local_model(local_model_path)
            return

        # check model exists in the hub first
        if not check_model_exists_on_hub(registry):
            raise ValueError(
                f"Model {registry} does not exist in the model-hub. Please check the registry at {MODEL_HUB_URL}."
            )

        # download model
        download_remote_model(registry, MODEL_CACHE_DIR)

        # load model once downloaded
        self._load_local_model(local_model_path)

    def _load_model(self):
        """
        Load the model from the path_to_model attribute.

        Order of operations:
        1. look for local model - this should be a path to a model.yaml
        2. look for model in cache
        3. download model from big.databio.org
        """
        # first just check if this is a local model
        if os.path.exists(self.path_to_model):
            self._load_local_model(self.path_to_model)
        # else, assume its on the model-hub
        else:
            self._load_remote_model(self.path_to_model)

    def convert_to_universe(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Converts the conesnsus peak set (.var) attributes of the AnnData object
        to a universe representation. This is done through interval overlap
        analysis.
        """
        # ensure adata has chr, start, and end
        if not all([x in adata.var.columns for x in ["chr", "start", "end"]]):
            raise ValueError(
                "AnnData object must have `chr`, `start`, and `end` columns in .var"
            )

        # create list of regions from adata
        query_set: List[str] = adata.var.apply(
            lambda x: f"{x['chr']}_{x['start']}_{x['end']}", axis=1
        ).tolist()
        universe_set = self.universe.regions

        # generate conversion map
        _map = generate_var_conversion_map(query_set, universe_set)

        # create a new DataFrame with the updated values
        updated_var = adata.var.copy()

        for i, row in tqdm(adata.var.iterrows(), total=adata.var.shape[0]):
            region = f"{row['chr']}_{row['start']}_{row['end']}"
            if region not in _map:
                continue

            # if it is, change the region to the universe region,
            # grab the first for now
            universe_region = _map[region][0]
            chr, start, end = universe_region.split("_")

            updated_var.at[i, "chr"] = chr
            updated_var.at[i, "start"] = start
            updated_var.at[i, "end"] = end

        # create a boolean mask of columns to keep
        columns_to_keep = np.array(
            [
                region in _map
                for region in updated_var.apply(
                    lambda x: f"{x['chr']}_{x['start']}_{x['end']}", axis=1
                )
            ]
        )

        # update adata with the new DataFrame and filtered columns
        adata = adata[:, columns_to_keep]
        adata.var = updated_var[columns_to_keep]
