import logging
import os
from typing import Dict, List, Any

import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import yacman
from tqdm import tqdm

from .const import MODEL_CACHE_DIR, MODEL_CONFIG_FILE_NAME, MODEL_HUB_URL, MODULE_NAME
from .models import ModelCard, Universe
from .utils import (
    anndata_to_regionsets,
    check_model_exists_on_hub,
    download_remote_model,
    generate_var_conversion_map,
    load_scembed_model,
    load_universe_file,
)

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
        Converts the consensus peak set (.var) attributes of the AnnData object
        to a universe representation. This is done through interval overlap
        analysis with bedtools.

        For each region in the `.var` attribute of the AnnData object, we
        either 1) map it to a region in the universe, or 2) map it to `None`.
        If it is mapped to `None`, it is not in the universe and will be dropped
        from the AnnData object. If it is mapped to a region, it will be updated
        to the region in the universe for downstream analysis.
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

        # find the regions that overlap with the universe
        # use dynamic programming to create a boolean mask of columns to keep
        columns_to_keep = []
        for i, row in tqdm(adata.var.iterrows(), total=adata.var.shape[0]):
            region = f"{row['chr']}_{row['start']}_{row['end']}"
            if _map[region] is None:
                columns_to_keep.append(False)
                continue

            # if it is, change the region to the universe region,
            # grab the first for now
            # TODO - this is a simplification, we should be able to handle multiple
            universe_region = _map[region]
            chr, start, end = universe_region.split("_")

            updated_var.at[i, "chr"] = chr
            updated_var.at[i, "start"] = start
            updated_var.at[i, "end"] = end

            columns_to_keep.append(True)

        # update adata with the new DataFrame and filtered columns
        adata = adata[:, columns_to_keep]
        adata.var = updated_var[columns_to_keep]

        return adata

    def get_embedding(self, region: str) -> np.ndarray:
        """
        Get the embedding for a region.
        """
        return self.model[region]

    def project(self, adata: sc.AnnData, key_added: str = "embedding") -> sc.AnnData:
        """
        Project the AnnData object into the model space. This is done in two steps:

        1. Convert the consensus peaks to a universe representation
        2. Project the universe representation into the model space using the
           model.

        :param adata: AnnData object to project
        :param key_added: Key to add to the .obsm attribute of the AnnData object
        """
        _LOGGER.info("Step 1/3: Converting consensus peaks to universe representation")
        adata_converted = self.convert_to_universe(adata)

        # convert each row to a region set
        _LOGGER.info("Step 2/3: Converting universe representation to region sets")
        region_sets = anndata_to_regionsets(adata_converted)

        # convert each region set to a vector by averaging the region
        # vectors in the model
        # TODO: what do we do if there are no region overlaps? i.e. the regionset
        # representation of the cell is []
        cell_embeddings = []
        _LOGGER.info("Step 3/3: Projecting region sets into model space")
        for region_set in tqdm(region_sets, total=len(region_sets)):
            embedding = np.mean(
                [
                    self.get_embedding(region)
                    for region in region_set
                    if region in self.model  # ignore regions not in the model
                ],
                axis=0,
            )
            # make sure not nan before appending,
            # otherwise append np.zeros
            if np.isnan(embedding).any():
                embedding = np.zeros(
                    self.model_config.model_parameters[0].embedding_dim
                )

            cell_embeddings.append(embedding)

        adata.obsm[key_added] = np.array(cell_embeddings)
        return adata

    def visualize_projection(
        self,
        universe: sc.AnnData,
        adata: sc.AnnData,
        embedding_key: str = "embedding",
        cluster_key: str = "leiden",
        n_rows: int = 2,
        plot_kwargs: Dict[str, Any] = {},
        fig_kwargs: Dict[str, Any] = {},
        color_palette: str = "tab20",
        random_state: int = 42,
    ):
        """
        Visualize the projection of the AnnData object into the model space.

        This requires that we have the original data the universe was trained on. In adition to that,
        we require that embeddings are attached to these sc.AnnData objects.  A small
        limitation that hopefully future versions will remove.

        :param universe: AnnData object of the universe
        :param adata: AnnData object to project
        :param embedding_key: Key in the .obsm attribute of the AnnData object that contains
                              the embedding
        :param cluster_key: Key in the .obs attribute of the universe that contains the
                            cluster information
        :param cluster_colors: Dictionary mapping cluster names to colors
        :param n_rows: Number of rows to use in the visualization (defaults to 2, but can be overridden if there are many clusters)
        :param n_cols: Number of columns to use in the visualization (defaults to 2, but can be overridden if there are many clusters)
        :param plot_kwargs: Additional keyword arguments to pass to the plot function of seaborn
        """
        # run umap on the universe
        _LOGGER.info("Running UMAP on the universe")
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        universe.obsm["X_umap"] = reducer.fit_transform(universe.obsm[embedding_key])
        universe.obsm["UMAP1"] = universe.obsm["X_umap"][:, 0]
        universe.obsm["UMAP2"] = universe.obsm["X_umap"][:, 1]

        # fit the adata to the universe umap
        _LOGGER.info("Fitting the adata to the universe UMAP")
        adata.obsm["X_umap"] = reducer.transform(adata.obsm[embedding_key])
        adata.obsm["UMAP1"] = adata.obsm["X_umap"][:, 0]
        adata.obsm["UMAP2"] = adata.obsm["X_umap"][:, 1]

        # check for cluster key
        if cluster_key not in adata.obs.columns:
            raise ValueError(
                f"AnnData object must have `{cluster_key}` in .obs to visualize"
            )

        # get dimensions of the plot
        n_clusters = len(adata.obs[cluster_key].unique())
        n_cols = int(np.ceil(n_clusters / n_rows))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), **fig_kwargs
        )
        fig.tight_layout(pad=3.0)

        # get cluster colors - color for each cluster
        cluster_colors = {
            cluster: color
            for cluster, color in zip(
                adata.obs[cluster_key].unique(), sns.color_palette(color_palette)
            )
        }

        # iterate over the clusters and plot accordingly
        for i, cluster in enumerate(adata.obs["leiden"].unique()):
            # get the axis to plot on
            ax_row = i // n_cols
            ax_col = i % n_cols
            ax = fig.axes[ax_row * n_cols + ax_col]

            # get color
            color = cluster_colors[cluster]

            # plot the universe
            sns.scatterplot(
                data=universe.obsm,
                x="UMAP1",
                y="UMAP2",
                color="gray",
                ax=ax,
                s=10,
                alpha=0.8,
                **plot_kwargs,
            )

            # plot the cluster on top
            sns.scatterplot(
                data=adata[adata.obs["leiden"] == cluster].obsm,
                x="UMAP1",
                y="UMAP2",
                color=color,
                ax=ax,
                s=10,
                alpha=1,
                **plot_kwargs,
            )

            # add legend to plot showing gray universe + colored cluster
            ax.legend(
                [
                    "Universe",
                    cluster,
                ],
                frameon=False,
                bbox_to_anchor=(0.5, -0.15),
                loc="upper center",
                ncol=2,
            )

            # set the title
            ax.set_title(f"Cluster {cluster}")

        return fig, axes
