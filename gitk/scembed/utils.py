import os
import shutil
import subprocess

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from logging import getLogger
from random import shuffle
from typing import TYPE_CHECKING, Dict, List, Union, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm

if TYPE_CHECKING:
    from .scembed import SCEmbed

from .const import *

_LOGGER = getLogger(MODULE_NAME)


class ScheduleType(Enum):
    """Learning rate schedule types"""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class LearningRateScheduler:
    """
    Simple class to track learning rates of the training procedure

    Based off of: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
    """

    def __init__(
        self,
        init_lr: float = DEFAULT_INIT_LR,
        min_lr: float = DEFAULT_MIN_LR,
        type: Union[str, ScheduleType] = ScheduleType.EXPONENTIAL,
        decay: float = None,
        n_epochs: int = None,
    ):
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.n_epochs = n_epochs

        # convert type to learning rate if necessary
        if isinstance(type, str):
            try:
                self.type = ScheduleType[type.upper()]
            except KeyError:
                raise ValueError(
                    f"Unknown schedule type: {type}. Must be one of ['linear', 'exponential']."
                )

        # init the current lr and iteration
        self._current_lr = init_lr
        self._iter = 1

        # init decay rate
        if decay is None:
            _LOGGER.warning(
                "No decay rate provided. Calculating decay rate from init_lr and n_epochs."
            )
            self.decay = init_lr / n_epochs
        else:
            self.decay = decay

    def _update_linear(self, epoch: int):
        lr = self.init_lr - (self.decay * epoch)
        return max(lr, self.min_lr)

    def _update_exponential(self, epoch: int):
        lr = self.get_lr() * (1 / (1 + self.decay * epoch))
        return max(lr, self.min_lr)

    def update(self):
        # update the learning rate according to the type
        if self.type == ScheduleType.LINEAR:
            self._current_lr = self._update_linear(self._iter)
            self._iter += 1
        elif self.type == ScheduleType.EXPONENTIAL:
            self._current_lr = self._update_exponential(self._iter)
            self._iter += 1
        else:
            raise ValueError(f"Unknown schedule type: {self.type}")

    def get_lr(self):
        return self._current_lr


class AnnDataChunker:
    def __init__(self, adata: sc.AnnData, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Simple class to chunk an AnnData object into smaller pieces. Useful for
        training on large datasets.

        :param adata: AnnData object to chunk. Must be in backed mode. See: https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html
        :param chunk_size: Number of cells to include in each chunk
        """
        self.adata = adata
        self.chunk_size = chunk_size
        self.n_chunks = len(adata) // chunk_size + 1

    def __iter__(self):
        for i in range(self.n_chunks):
            chunk = self.adata[i * self.chunk_size : (i + 1) * self.chunk_size, :]
            yield chunk.to_memory()

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, item):
        return self.adata[item * self.chunk_size : (item + 1) * self.chunk_size, :]

    def __repr__(self):
        return f"<AnnDataChunker: {self.n_chunks} chunks of size {self.chunk_size}>"


def shuffle_documents(
    documents: List[List[str]],
    n_shuffles: int,
    threads: int = None,
) -> List[List[str]]:
    """
    Shuffle around the genomic regions for each cell to generate a "context".

    :param List[List[str]] documents: the document list to shuffle.
    :param int n_shuffles: The number of shuffles to conduct.
    :param int threads: The number of threads to use for shuffling.
    """

    def shuffle_list(l: List[str], n: int) -> List[str]:
        for _ in range(n):
            shuffle(l)
        return l

    _LOGGER.debug(f"Shuffling documents {n_shuffles} times.")
    shuffled_documents = documents.copy()
    with ThreadPoolExecutor(max_workers=threads) as executor:
        shuffled_documents = list(
            tqdm(
                executor.map(
                    shuffle_list,
                    shuffled_documents,
                    [n_shuffles] * len(documents),
                ),
                total=len(documents),
            )
        )
    return shuffled_documents


def remove_regions_below_min_count(
    region_sets: List[List[str]], min_count: int
) -> List[List[str]]:
    """
    Remove regions that don't satisfy the min count.

    TODO - this is a bit slow, could be sped up using dataframe operations.

    :param List[List[str]] region_sets: the region sets to remove regions from
    :param int min_count: the min count to use
    """
    # get the counts for each region
    region_counts = Counter()
    for region_set in region_sets:
        region_counts.update(region_set)

    # remove any regions that dont satisfy the min count
    region_sets = [
        [r for r in region_set if region_counts[r] >= min_count] for region_set in region_sets
    ]

    return region_sets


def load_scanpy_data(path_to_h5ad: str) -> sc.AnnData:
    """
    Load in the h5ad file that holds all of the information
    for our single-cell data with scanpy.

    :param str path_to_h5ad: the path to the h5ad file made with scanpy
    """
    return sc.read_h5ad(path_to_h5ad)


def extract_region_list(region_df: pd.DataFrame) -> List[str]:
    """
    Parse the `var` attribute of the scanpy.AnnData object and
    return a list of regions from the matrix. Converts each
    region to a string of the form `chr_start_end`.

    :param pandas.DataFrame region_df: the regions dataframe to parse
    """
    _LOGGER.info("Extracting region list from matrix.")
    regions = [
        f"{row[CHR_KEY]}_{row[START_KEY]}_{row[END_KEY]}" for _, row in region_df.iterrows()
    ]
    return regions


def convert_anndata_to_documents(
    anndata: sc.AnnData,
    use_defaults: bool = True,
) -> List[List[str]]:
    """
    Parses the scanpy.AnnData object to create the required "documents" object for
    training the Word2Vec model. Each row (or cell) is treated as a "document". That
    is, each region is a "word", and the total collection of regions is the "document".

    :param scanpy.AnnData anndata: the AnnData object to parse.
    :use_defaults bool: whether or not to use the default column names for the regions. this
                        is most commonly used when the AnnData object was created without using
                        the var attribute.
    """
    # enable progress bar
    tqdm.pandas()

    if use_defaults:
        regions_parsed = [f"r{i}" for i in range(anndata.var.shape[0])]
        # drop var attribute since it messes with things
        anndata.var.drop(anndata.var.columns, axis=1, inplace=True)
        anndata.var.reset_index(inplace=True)

        # add the region column back in
        anndata.var["region"] = regions_parsed
    else:
        regions_parsed = extract_region_list(anndata.var)
    sc_df = anndata.to_df()
    _LOGGER.info("Generating documents.")

    docs = [
        [
            regions_parsed[int(region_indx)]
            for region_indx, value in row.to_dict().items()
            if value > 0
        ]
        for _, row in sc_df.iterrows()
    ]

    return docs


def load_scembed_model(path: str) -> "SCEmbed":
    """
    Load a scembed model from disk.

    :param str path: The path to the model.
    """
    import pickle
    import gzip

    with gzip.open(path, "rb") as f:
        model = pickle.load(f)
        return model


def load_scembed_model_deprecated(path: str) -> "SCEmbed":
    """
    Load a scembed model from disk. THIS IS DEPRECATED AND WILL BE REMOVED IN THE FUTURE.

    :param str path: The path to the model.
    """
    import pickle

    with open(path, "rb") as f:
        model = pickle.load(f)
        return model


def check_model_exists_on_hub(registry: str) -> bool:
    """
    Check the model hub for the existing model registry. Registry
    is of the name <namespace>/<model_name>.

    Looks for the existence of model.yaml at {MODEL_HUB_URL}/{registry_name}/model.yaml
    """
    # check if model exists in the hub
    url = f"{MODEL_HUB_URL}/{registry}/model.yaml"
    cmd = f"curl -s -o /dev/null -w '%{{http_code}}' {url}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8") == "200"


def download_remote_model(registry: str, path: str, overwrite: bool = True) -> None:
    """
    Download a model from the model hub. Overwrites any existing.

    :param str registry: the registry name of the model to download
    :param str path: the path to download the model to, this is the folder that will contain registry/model.yaml
    """
    path_to_model = os.path.join(path, registry)
    if os.path.exists(path_to_model) and overwrite:
        _LOGGER.debug("Removing existing model.")
        shutil.rmtree(path_to_model)

    cmd = f"wget -r -np -nH --cut-dirs=2 -P {path} -l 1 {MODEL_HUB_URL}/{registry}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    if result.returncode != 0:
        raise ValueError("Could not download model from model-hub.")


def load_universe_file(path: str) -> List[str]:
    """
    Loads a bed file into a list of regions. Assumes
    the bed file is of the form chr start end (tab-separated).
    """
    with open(path, "r") as f:
        regions = [line.strip().split("\t") for line in f.readlines()]
        regions = [f"{r[0]}_{r[1]}_{r[2]}" for r in regions]
    return regions


def generate_var_conversion_map(
    a: List[str],
    b: List[str],
    path_to_bedtools: str = None,
    fraction: float = 1.0e-9,
) -> Dict[str, Union[str, None]]:
    """
    Create a conversion map to convert regions from a to b. This is used to convert the
    consensus peak set of one AnnData object to another.

    For each region in a, we will either find a matching region in b, or None. If a matching
    region is found, we will store the region in b. If no matching region is found, we will
    store `None`.

    Intuitively, think of this as converting `A` --> `B`. If a region in `A` is found in `B`,
    we will change the region in `A` to the region in `B`. If a region in `A` is not found in
    `B`, we will drop that region in `A` altogether.

    :param List[str] a: the first list of regions
    :param List[str] b: the second list of regions
    :param str path_to_bedtools: the path to the bedtools executable
    :param float fraction: the fraction of the region that must overlap to be considered an overlap
    """
    # write a and b to temp files in cache
    a_file = os.path.join(MODEL_CACHE_DIR, "a.bed")
    b_file = os.path.join(MODEL_CACHE_DIR, "b.bed")

    # write a and b to temp files
    with open(a_file, "w") as f:
        # split each region into chr start end
        a_parsed = [region.split("_") for region in a]
        a_parsed = [f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in a_parsed]
        f.writelines(a_parsed)
    with open(b_file, "w") as f:
        b_parsed = [region.split("_") for region in b]
        b_parsed = [f"{r[0]}\t{r[1]}\t{r[2]}\n" for r in b_parsed]
        f.writelines(b_parsed)

    # sort both files
    cmd = f"sort -k1,1 -k2,2n {a_file} -o {a_file}"
    subprocess.run(cmd, shell=True)

    cmd = f"sort -k1,1 -k2,2n {b_file} -o {b_file}"
    subprocess.run(cmd, shell=True)

    # run bedtools
    bedtools_cmd = f"intersect -a {a_file} -b {b_file} -wa -wb -f {fraction}"

    # add path to bedtools if provided
    if path_to_bedtools is not None:
        cmd = f"{path_to_bedtools} {bedtools_cmd}"
    else:
        cmd = f"bedtools {bedtools_cmd}"

    # target file
    target_file = os.path.join(MODEL_CACHE_DIR, "olaps.bed")
    with open(target_file, "w") as f:
        subprocess.run(cmd, shell=True, stdout=f)

    # bedtools reports overlaps like this:
    # chr1 100 200 chr1 150 250
    # we want to convert this to a map like this:
    # {chr1_100_200: chr1_150_250}
    # we will use a dictionary to do this
    # if a region in A overlaps with multiple regions in B, we will
    # take the first one. as such we need to check if a region in A
    # has already been mapped to a region in B
    conversion_map = dict()
    with open(target_file, "r") as f:
        for line in f.readlines():
            line = line.strip().split("\t")
            a_region = f"{line[0]}_{line[1]}_{line[2]}"
            b_region = f"{line[3]}_{line[4]}_{line[5]}"
            if a_region not in conversion_map:
                conversion_map[a_region] = b_region

    # add `None` mappings for regions in A that did not overlap with any regions in B
    for region in a:
        if region not in conversion_map:
            conversion_map[region] = None

    # remove temp files
    os.remove(a_file)
    os.remove(b_file)
    os.remove(target_file)

    return conversion_map


def anndata_to_regionsets(adata: sc.AnnData) -> List[List[str]]:
    """
    Converts an AnnData object to a list of lists of regions. This
    is done by taking each cell and creating a list of all regions
    that have a value greater than 0.

    *Note: this method requires that the sc.AnnData object have
    chr, start, and end in `.var` attributes*
    """
    # Extract the arrays for chr, start, and end
    chr_values = adata.var["chr"].values
    start_values = adata.var["start"].values
    end_values = adata.var["end"].values

    # Perform the comparison using numpy operations
    positive_values = adata.X > 0

    if not isinstance(positive_values, np.ndarray):
        positive_values = positive_values.toarray()

    regions = []
    for i in tqdm(range(adata.shape[0]), total=adata.shape[0]):
        regions.append(
            [
                f"{chr_values[j]}_{start_values[j]}_{end_values[j]}"
                for j in np.where(positive_values[i])[0]
            ]
        )
    return regions


def barcode_mtx_peaks_to_anndata(
    barcodes_path: str,
    mtx_path: str,
    peaks_path: str,
    transpose: bool = True,
    sparse: bool = True,
) -> sc.AnnData:
    """
    This function will take three files:

    1. a barcodes file (.tsv)
    2. a matrix file (.mtx)
    3. a peaks file (.bed)

    And turn them into an AnnData object. It will attach the peaks
    as a .var attribute (chr, start, end) and the barcodes as a .obs attribute (index)

    :param str barcodes_path: the path to the barcodes file
    :param str mtx_path: the path to the matrix file
    :param str peaks_path: the path to the peaks file
    :param bool transpose: whether or not to transpose the matrix
    :param bool sparse: whether or not the matrix is sparse

    :return sc.AnnData: an AnnData object
    """
    from scipy.io import mmread

    # load the barcodes
    barcodes = pd.read_csv(barcodes_path, sep="\t", header=None)
    barcodes.columns = ["barcode"]
    barcodes.index = barcodes["barcode"]

    # load the mtx
    mtx = mmread(mtx_path)
    if transpose:
        mtx = mtx.T

    # load the peaks
    peaks = pd.read_csv(peaks_path, sep="\t", header=None)
    peaks.columns = ["chr", "start", "end"]

    # create the AnnData object
    adata = sc.AnnData(X=mtx, obs=barcodes, var=peaks)

    return adata


def create_model_info_dict(
    path_to_weights: Optional[str] = None,
    path_to_universe: Optional[str] = None,
    name: Optional[str] = None,
    reference: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    model_parameters: Optional[Dict[str, Union[str, int]]] = None,
    model_architecture: Optional[str] = None,
    maintainers: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Union[str, List[str], List[Dict[str, Union[str, int]]]]]:
    model_info = {}

    if path_to_weights:
        model_info["path_to_weights"] = path_to_weights
    if path_to_universe:
        model_info["path_to_universe"] = path_to_universe
    if name:
        model_info["name"] = name
    if reference:
        model_info["reference"] = reference
    if description:
        model_info["description"] = description
    if tags:
        model_info["tags"] = tags
    if datasets:
        model_info["datasets"] = datasets
    if model_parameters:
        model_info["model_parameters"] = model_parameters
    if model_architecture:
        model_info["model_architecture"] = model_architecture
    if maintainers:
        model_info["maintainers"] = maintainers

    return model_info
