import multiprocessing
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np
import scanpy as sc
from intervaltree import IntervalTree
from tqdm import tqdm

import gitk.region2vec.utils as utils
from gitk.tokenization.split_file import split_file

from ..io import Region, RegionSet
from .hard_tokenization_batch import main as hard_tokenization
from .utils import anndata_to_regionsets


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError


class InMemTokenizer(Tokenizer):
    """Abstract class representing a tokenizer function"""

    def __init__(self, universe: Union[str, RegionSet, None] = None):
        """
        Create a new InMemTokenizer.

        You can either pass a RegionSet object or a path to a BED file containing regions.

        :param Union[str, RegionSet] universe: The universe to use for tokenization.
        """
        self._trees: Dict[str, IntervalTree] = dict()

        if isinstance(universe, str):
            self.universe = RegionSet(universe)  # load from file
        elif isinstance(universe, RegionSet):
            self.universe = universe  # load from memory (probably rare?)
        else:
            self.universe = None

        if self.universe is not None:
            # build interval trees
            self.build_trees()

    def get_tree(self, tree: str):
        """
        Get the interval tree for a given chromosome.

        :param str tree: The chromosome to get the tree for.
        """
        return self._trees[tree]

    @property
    def trees(self):
        return self._trees

    def add_regions(
        self, regions: Union[str, List[Region], RegionSet, List[RegionSet], List[List[Region]]]
    ):
        """
        Add regions to the universe.
        """
        self.build_trees(regions)

    def build_trees(
        self,
        regions: Union[
            str, List[str], List[Region], RegionSet, List[RegionSet], List[List[Region]]
        ] = None,
    ):
        """
        Builds the interval tree from the regions. The tree is a dictionary that maps chromosomes to
        interval trees.

        We read in the regions (either from memory or from a BED file) and convert them to an
        interval tree.

        :param str regions: The regions to use for tokenization. This can be thought of as a vocabulary.
        """
        # just return if we already have a universe
        if regions is None:
            regions = self.universe  # use the universe passed in the constructor
        # check for bed file
        elif isinstance(regions, str):
            regions = RegionSet(regions)  # load from file
        # check for list of bed files
        elif isinstance(regions, list) and isinstance(regions[0], str):
            regions = []
            for bedfile in regions:
                regions.extend([region for region in RegionSet(bedfile)])
        # check for regionset
        elif isinstance(regions, RegionSet):
            pass
        # check for list of regionsets
        elif isinstance(regions, list) and isinstance(regions[0], RegionSet):
            regions = [region for regionset in regions for region in regionset]
        # check for list of lists of regions
        elif isinstance(regions, list) and isinstance(regions[0], list):
            regions = [region for regionlist in regions for region in regionlist]
        # check for list of regions
        elif isinstance(regions, list) and isinstance(regions[0], Region):
            pass

        # build trees
        for region in tqdm(regions, total=len(regions), desc="Adding regions to universe"):
            if region.chr not in self._trees:
                self._trees[region.chr] = IntervalTree()
            self._trees[region.chr][
                region.start : region.end
            ] = f"{region.chr}_{region.start}_{region.end}"

        # count total regions
        self.total_regions = sum([len(self._trees[tree]) for tree in self._trees])

    def fit(
        self, regions: Union[str, List[Region], RegionSet, List[RegionSet], List[List[Region]]]
    ):
        """
        Fit the tokenizer to the given regions. This is equivalent to adding regions to the universe.
        """
        self.add_regions(regions)

    def find_overlaps(
        self,
        regions: Union[str, List[Region], Region, RegionSet],
        f: float = None,  # not implemented yet
    ) -> List[Region]:
        """
        Query the interval tree for the given regions. That is, find all regions that overlap with the given regions.

        :param Union[str, List[str], List[Region], Region, RegionSet] regions: The regions to query for. this can be either a bed file,
                                                                            a list of regions (chr_start_end), or a list of tuples of chr, start, end.
        :param float f: The fraction of the region that must overlap to be considered an overlap. Not yet implemented.
        """
        # validate input
        if isinstance(regions, Region):
            regions = [regions]
        elif isinstance(regions, str):
            regions = RegionSet(regions)
        else:
            pass

        overlapping_regions = []
        for region in regions:
            if region.chr not in self._trees:
                continue
            overlaps = self._trees[region.chr][region.start : region.end]
            for overlap in overlaps:
                overlapping_regions.append(Region(region.chr, overlap.begin, overlap.end))
        return overlapping_regions

    def convert_anndata_to_universe(self, adata: sc.AnnData) -> sc.AnnData:
        """
        Convert an AnnData object to a universe. This is done by
        inspecting the peaks in the AnnData object and converting
        them to regions found in the universe. This process
        occurs in-place.

        :param sc.AnnData adata: The AnnData object to convert.
        :param Universe universe: The universe to convert to.
        """
        # ensure adata has chr, start, and end
        if not all([x in adata.var.columns for x in ["chr", "start", "end"]]):
            raise ValueError("AnnData object must have `chr`, `start`, and `end` columns in .var")

        # create list of regions from adata
        adata.var["region"] = (
            adata.var["chr"].astype(str)
            + "_"
            + adata.var["start"].astype(str)
            + "_"
            + adata.var["end"].astype(str)
        )
        query_set = [
            Region(x[0], x[1], x[2])
            for x in list(
                zip(adata.var["chr"], adata.var["start"].astype(int), adata.var["end"].astype(int))
            )
        ]

        # generate conversion map
        _map = self.generate_var_conversion_map(query_set)

        # map regions to new universe
        adata.var["new_region"] = adata.var["region"].map(_map)

        # drop rows where new_region is None
        adata.var.dropna(subset=["new_region"], inplace=True)

        # split new_region into chr, start, end
        adata.var[["chr", "start", "end"]] = adata.var["new_region"].str.split("_", expand=True)

        # drop 'region' and 'new_region' columns
        adata.var.drop(columns=["region", "new_region"], inplace=True)

        # update adata with the new DataFrame
        adata = adata[:, adata.var.index]

        return adata

    def generate_var_conversion_map(
        self,
        a: List[Region],
        fraction: float = 1.0e-9,  # not used
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

        :param List[tuple[str, int, int]] a: the first list of regions
        :param Universe: the second list of regions as a Universe object
        :param float fraction: the fraction of the region that must overlap to be considered an overlap. Not used.
        """

        conversion_map = dict()

        for region in tqdm(a, total=len(a), desc="Generating conversion map"):
            overlaps = self.find_overlaps(region)
            region_str = f"{region.chr}_{region.start}_{region.end}"
            if len(overlaps) > 0:
                olap = overlaps[0]  # take the first overlap for now, we can change this later
                olap_str = f"{olap.chr}_{olap.start}_{olap.end}"
                conversion_map[region_str] = olap_str
            else:
                conversion_map[region_str] = None

        return conversion_map

    def tokenize(
        self, region_set: Union[str, List[Region], RegionSet, sc.AnnData]
    ) -> Union[List[Region], List[List[Region]], List[RegionSet]]:
        """
        Tokenize a RegionSet.

        This is achieved using hard tokenization which is just a query to the universe, or
        simple overlap detection.

        :param str | List[Region] | sc.AnnData region_set: The list of regions to tokenize
        """
        if isinstance(region_set, sc.AnnData):
            # step 1 is to convert the AnnData object to the universe
            region_set = self.convert_anndata_to_universe(region_set)

            # step 2 is to convert the AnnData object to a list of lists of regions
            return anndata_to_regionsets(region_set)
        else:
            return self.find_overlaps(region_set)

    # not sure what this looks like, multiple RegionSets?
    def tokenize_rsc(self, rsc: RegionSetCollection) -> RegionSetCollection:
        """Tokenize a RegionSetCollection"""
        raise NotImplementedError


class FileTokenizer(Tokenizer):
    """Tokenizer that tokenizes files"""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def tokenize(input_globs: List[str], output_folder: str, universe_path: str):
        raise NotImplementedError


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def hard_tokenization_main(
    src_folder: str,
    dst_folder: str,
    universe_file: str,
    fraction: float = 1e-9,
    file_list: list[str] = None,
    num_workers: int = 10,
    bedtools_path: str = "bedtools",
) -> int:
    """Tokenizes raw BED files in parallel.

    This is the main function for hard tokenization. It uses multiple processes to
    speed up the tokenization process.

    Args:
        src_folder (str): The folder where raw BED files reside.
        dst_folder (str): The foder to store tokenized BED files.
        universe_file (str): The path to a universe file.
        fraction (float, optional): A parameter for bedtools.intersect.
            Defaults to 1e-9.
        file_list (list[str], optional): A list of files (just names not full
            paths) that need to be tokenized. Defaults to None and uses all BED
            files in src_folder.
        num_workers (int, optional): Number of processes used. Defaults to 10.
        bedtools_path (str, optional): The path to a bedtools binary. Defaults
            to "bedtools".

    Raises:
        Exception: No bedtools executable found

    Returns:
        int: 0 when the dst_folder folder has incomplete list of tokenized BED
            files which should be removed first; 1 when the dst_folder folder
            has the complete tokenized BED files or the tokenization process
            succeeds.
    """
    timer = utils.Timer()
    start_time = timer.t()

    file_list_path = os.path.join(dst_folder, "file_list.txt")
    files = os.listdir(src_folder)
    file_count = len(files)
    if file_count == 0:
        print(f"No files in {src_folder}")
        return 0

    os.makedirs(dst_folder, exist_ok=True)
    if file_list is None:  # use all bed files in data_folder
        # generate a file list
        file_list = files
        print(f"Use all ({file_count}) bed files in {src_folder}")
    else:
        file_number = len(file_list)
        print(f"{file_count} bed files in total, use {file_number} of them")

    # check whether all files in file_list are tokenized
    number = -1
    if os.path.exists(dst_folder):
        all_set = set([f.strip() for f in file_list])
        existing_set = set(os.listdir(dst_folder))
        not_covered = all_set - existing_set
        number = len(not_covered)
    if number == 0 and len(existing_set) == len(all_set):
        print("Skip tokenization. Using the existing tokenization files")
        return 1
    elif len(existing_set) > 0:
        print(
            f"Folder {dst_folder} exists with incomplete tokenized files. Please empty/delete the folder first"
        )
        return 0

    with open(file_list_path, "w") as f:
        for file in file_list:
            f.write(file)
            f.write("\n")

    if bedtools_path == "bedtools":
        try:
            rval = subprocess.call([bedtools_path, "--version"])
        except:
            raise Exception("No bedtools executable found")
        if rval != 0:
            raise Exception("No bedtools executable found")

    print(f"Tokenizing {len(file_list)} bed files ...")

    file_count = len(file_list)
    # split the file_list into several subsets for each worker to process in parallel
    nworkers = min(int(np.ceil(file_count / 20)), num_workers)
    if nworkers <= 1:
        tokenization_args = Namespace(
            data_folder=src_folder,
            file_list=file_list_path,
            token_folder=dst_folder,
            universe=universe_file,
            bedtools_path=bedtools_path,
            fraction=fraction,
        )
        hard_tokenization(tokenization_args)

    else:  # multiprocessing
        dest_folder = os.path.join(dst_folder, "splits")
        split_file(file_list_path, dest_folder, nworkers)
        args_arr = []
        for n in range(nworkers):
            temp_token_folder = os.path.join(dst_folder, f"batch_{n}")
            tokenization_args = Namespace(
                data_folder=src_folder,
                file_list=os.path.join(dest_folder, f"split_{n}.txt"),
                token_folder=temp_token_folder,
                universe=universe_file,
                bedtools_path=bedtools_path,
                fraction=fraction,
            )
            args_arr.append(tokenization_args)
        with multiprocessing.Pool(nworkers) as pool:
            processes = [pool.apply_async(hard_tokenization, args=(param,)) for param in args_arr]
            results = [r.get() for r in processes]
        # move tokenized files in different folders to expr_tokens
        shutil.rmtree(dest_folder)
        for param in args_arr:
            allfiles = os.listdir(param.token_folder)
            for f in allfiles:
                shutil.move(
                    os.path.join(param.token_folder, f),
                    os.path.join(dst_folder, f),
                )
            shutil.rmtree(param.token_folder)
    os.remove(file_list_path)
    print(f"Tokenization complete {len(os.listdir(dst_folder))}/{file_count} bed files")
    elapsed_time = timer.t() - start_time
    print(f"[Tokenization] {utils.time_str(elapsed_time)}/{utils.time_str(timer.t())}")
    return 1
