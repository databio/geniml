import multiprocessing
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import genomicranges as gr
import numpy as np
import pandas as pd
import scanpy as sc
from gtokenizers import (
    Region as GRegion,
    TreeTokenizer as GTreeTokenizer,
    TokenizedRegionSet as GTokenizedRegionSet,
    Universe as GUniverse,
)
from intervaltree import IntervalTree
from rich.progress import track

from geniml.tokenization.split_file import split_file

from ..io import Region, RegionSet, RegionSetCollection
from .hard_tokenization_batch import main as hard_tokenization
from .utils import anndata_to_regionsets, time_str, Timer
from ..utils import wordify_region


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError


class ITTokenizer(Tokenizer):
    """
    A fast, in memory, tokenizer that uses `gtokenizers` - a rust based tokenizer. This
    tokenizer is the fastest tokenizer available. It is also the most memory efficient.

    It should be a near dropin replacement for InMemTokenizer.
    """

    # class based method to insantiate the tokenizer from file
    @classmethod
    def from_file(cls, universe: str):
        """
        Create a new tokenizer from a file.

        Usage:
        ```
        tokenizer = ITTokenizer.from_file("path/to/universe.bed")
        ```

        :param str universe: The universe to use for tokenization.
        """
        return cls(universe)

    @property
    def universe(self) -> GUniverse:
        return self._tokenizer.universe

    def __init__(self, universe: str = None):
        """
        Create a new tokenizer.

        This tokenizer only accepts a path to a BED file containing regions.

        :param str universe: The universe to use for tokenization.
        """
        if universe is not None:
            self._tokenizer: GTreeTokenizer = GTreeTokenizer(universe)
        else:
            self._tokenizer = None

    def tokenize(self, query: Union[Region, RegionSet]) -> GTokenizedRegionSet:
        """
        Tokenize a Region or RegionSet into the universe

        :param Union[Region, RegionSet] query: The query to tokenize.
        """
        if isinstance(query, Region):
            query = [query]
        elif isinstance(query, RegionSet):
            query = list(query)
        elif isinstance(query, list) and isinstance(query[0], Region):
            pass
        else:
            raise ValueError("Query must be a Region or RegionSet")

        result = self._tokenizer.tokenize(list(query))
        return result

    def padding_token(self) -> Region:
        return self._tokenizer.padding_token

    def padding_token_id(self) -> int:
        padding_token = self.padding_token()
        return self.universe.region_to_id(
            GRegion(padding_token.chr, padding_token.start, padding_token.end)
        )

    def convert_tokens_to_ids(self, tokens: GTokenizedRegionSet) -> List[int]:
        """
        Convert a list of tokens to a list of ids.

        :param List[TokenizedRegion] tokens: The list of tokens to convert
        """
        return [token.id for token in tokens]

    def __len__(self):
        return len(self._tokenizer)


class InMemTokenizer(Tokenizer):
    """
    Tokenize new regions into a vocabulary.

    This is done using hard tokenization which is just a query to the universe, or
    simple overlap detection. Computation occurs in memory. This is the fastest
    tokenizer, but it is not scalable to large universes.
    """

    def __init__(self, universe: Union[str, RegionSet, None] = None):
        """
        Create a new InMemTokenizer.

        You can either pass a RegionSet object or a path to a BED file containing regions.

        :param Union[str, RegionSet] universe: The universe to use for tokenization.
        """
        self._trees: Dict[str, IntervalTree] = dict()
        self._region_to_index: Dict[str, int] = dict()

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
    def trees(self) -> Dict[str, IntervalTree]:
        return self._trees

    @property
    def region_to_index(self) -> Dict[str, int]:
        return self._region_to_index

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

        # build trees + add regions to universe + make region to index map
        self.universe = regions
        indx = 0
        for region in track(regions, total=len(regions), description="Adding regions to universe"):
            # r_string = wordify_region(region)
            if region.chr not in self._trees:
                self._trees[region.chr] = IntervalTree()
            self._trees[region.chr][region.start : region.end] = None

            # # add to region to index map
            # if r_string not in self._region_to_index:
            #     self._region_to_index[r_string] = indx
            #     indx += 1

        # count total regions
        self.total_regions = sum([len(self._trees[tree]) for tree in self._trees])

    def find_overlaps(
        self,
        regions: Union[str, List[Region], Region, RegionSet],
        f: float = None,  # not implemented yet,
        return_all: bool = False,
    ) -> List[Region]:
        """
        Query the interval tree for the given regions. That is, find all regions that overlap with the given regions.

        :param Union[str, List[str], List[Region], Region, RegionSet] regions: The regions to query for. this can be either a bed file,
                                                                            a list of regions (chr_start_end), or a list of tuples of chr, start, end.
        :param float f: The fraction of the region that must overlap to be considered an overlap. Not yet implemented.
        :param bool return_all: Whether to return all overlapping regions or just the first. Defaults to False. (in the future, we can change this to return the top k or best)
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
                print(f"Warning: Could not find {region.chr} in universe.")
                continue

            overlaps = self._trees[region.chr][region.start : region.end]
            if not overlaps:
                overlapping_regions.append(None)
            else:
                overlaps = list(overlaps)
                if return_all:
                    overlapping_regions.extend(
                        [Region(region.chr, olap.begin, olap.end) for olap in overlaps]
                    )
                else:
                    olap = overlaps[0]
                    overlapping_regions.append(Region(region.chr, olap.begin, olap.end))

        return overlapping_regions

    def __len__(self, *args, **kwargs):
        if self.universe is None:
            return 0
        else:
            return len(self.universe)

    def convert_anndata_to_universe(self, adata: sc.AnnData) -> sc.AnnData:
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
            raise ValueError("AnnData object must have `chr`, `start`, and `end` columns in .var")

        # create list of regions from adata
        query_set: List[tuple[str, int, int]] = adata.var.apply(
            lambda x: Region(x["chr"], int(x["start"]), int(x["end"])), axis=1
        ).tolist()

        # generate conversion map
        _map = self.generate_var_conversion_map(query_set)

        # create a new DataFrame with the updated values
        updated_var = adata.var.copy()

        # find the regions that overlap with the universe
        # use dynamic programming to create a boolean mask of columns to keep
        columns_to_keep = []
        for i, row in track(
            adata.var.iterrows(), total=adata.var.shape[0], description="Converting to universe"
        ):
            region = f"{row['chr']}_{row['start']}_{row['end']}"
            if _map[region] is None:
                columns_to_keep.append(False)
                continue

            # if it is, change the region to the universe region,
            # grab the first for now
            # TODO - this is a simplification, we should be able to handle multiple
            universe_region = _map[region]
            try:
                chr, start, end = universe_region.split("_")
                start = int(start)
                end = int(end)
            except ValueError:
                raise ValueError(f"Could not parse region {universe_region}")

            updated_var.at[i, "chr"] = chr
            updated_var.at[i, "start"] = start
            updated_var.at[i, "end"] = end

            columns_to_keep.append(True)

        # update adata with the new DataFrame and filtered columns
        adata = adata[:, columns_to_keep]
        adata.var = updated_var[columns_to_keep]

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

        for region in track(a, total=len(a), description="Generating conversion map"):
            overlaps = self.find_overlaps(region)
            region_str = f"{region.chr}_{region.start}_{region.end}"
            overlaps = [olap for olap in overlaps if olap is not None]
            if len(overlaps) > 0:
                olap = overlaps[0]  # take the first overlap for now, we can change this later
                olap_str = f"{olap.chr}_{olap.start}_{olap.end}"
                conversion_map[region_str] = olap_str
            else:
                conversion_map[region_str] = None

        return conversion_map

    def tokenize(
        self, regions: Union[str, List[Region], RegionSet, sc.AnnData], return_all: bool = True
    ) -> Union[List[Region], List[List[Region]], List[RegionSet]]:
        """
        Tokenize a RegionSet.

        This is achieved using hard tokenization which is just a query to the universe, or
        simple overlap detection.

        :param str | List[Region] | sc.AnnData regions: The list of regions to tokenize
        :param bool return_all: Whether to return all overlapping regions or just the first. Defaults to True. (in the future, we can change this to return the top k or best)
                                Note that returning all might return more than one region per region in the input. Thus, you lose the 1:1 mapping.
        """
        if isinstance(regions, sc.AnnData):
            # step 1 is to convert the AnnData object to the universe
            regions = self.convert_anndata_to_universe(regions)

            # step 2 is to convert the AnnData object to a list of lists of regions
            return anndata_to_regionsets(regions)
        else:
            return self.find_overlaps(regions, return_all=return_all)

    def convert_tokens_to_ids(
        self, tokens: List[Region], missing_token_id: any = None
    ) -> List[int]:
        """
        Convert a list of tokens to a list of ids.

        :param List[Region] tokens: The list of tokens to convert
        """
        return [
            missing_token_id if token is None else self._region_to_index[wordify_region(token)]
            for token in tokens
        ]

    def tokenize_and_convert_to_ids(
        self, regions: Union[str, List[Region], RegionSet, sc.AnnData], return_all: bool = False
    ) -> List[int]:
        """
        Tokenize a RegionSet and convert to ids.

        This is achieved using hard tokenization which is just a query to the universe, or
        simple overlap detection.

        :param str | List[Region] | sc.AnnData regions: The list of regions to tokenize
        :param bool return_all: Whether to return all overlapping regions or just the first. Defaults to False. (in the future, we can change this to return the top k or best)
                                Note that returning all might return more than one region per region in the input. Thus, you lose the 1:1 mapping.
        """
        tokens = self.tokenize(regions, return_all=return_all)
        return self.convert_tokens_to_ids(tokens)

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
    timer = Timer()
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
    print(f"[Tokenization] {time_str(elapsed_time)}/{time_str(timer.t())}")
    return 1
