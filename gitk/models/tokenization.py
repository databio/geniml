from typing import List, Union, Dict

import scanpy as sc
import os

from intervaltree import IntervalTree

from .utils import (
    validate_region_input,
    convert_to_universe,
    anndata_to_regionsets,
)


class Universe:
    def __init__(self, regions: Union[str, List[str]]):
        """
        Create a new Universe.

        :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                        This can be either a list of regions or a path to a BED file containing regions.
        """
        self._trees: Dict[str, IntervalTree] = dict()
        self._regions = regions
        self._universe_set: list[tuple[str, int, int]] = []

        if regions is not None:
            self.build_tree()
            self._build_universe_set()

    def get_tree(self, tree: str):
        return self._trees[tree]

    @property
    def trees(self):
        return self._trees

    @property
    def regions(self):
        return self._regions

    @property
    def universe_set(self):
        return self._universe_set

    def _build_universe_set(self):
        """
        Return the universe as a set of regions in the form (chr, start, end).
        """
        universe = []
        for tree in self._trees:
            for interval in self._trees[tree]:
                universe.append((tree, interval.begin, interval.end))
        self._universe_set = universe

    def build_tree(self, regions: str = None):
        """
        Builds the interval tree from the regions. The tree is a dictionary that maps chromosomes to
        interval trees.

        We read in the regions (either from memory or from a BED file) and convert them to an
        interval tree.
        """
        regions = validate_region_input(regions or self._regions)

        # build trees
        for region in regions:
            chr, start, end = region
            start = int(start)
            end = int(end)
            if chr not in self._trees:
                self._trees[chr] = IntervalTree()
            self._trees[chr][start:end] = f"{chr}_{start}_{end}"

    def query(
        self,
        regions: Union[
            str, List[str], List[tuple[str, int, int]], tuple[str, int, int]
        ],
    ):
        """
        Query the interval tree for the given regions.

        :param regions: The regions to query for. this can be either a bed file,
                        a list of regions (chr_start_end), or a list of tuples of chr, start, end.
        """
        # validate input
        if isinstance(regions, tuple):
            regions = [regions]
        regions = validate_region_input(regions)

        overlapping_regions = []
        for region in regions:
            chr, start, end = region
            start = int(start)
            end = int(end)
            if chr not in self._trees:
                continue
            overlaps = self._trees[chr][start:end]
            for overlap in overlaps:
                overlapping_regions.append((chr, overlap.begin, overlap.end))
        return overlapping_regions

    def __contains__(self, item: tuple[str, int, int]):
        # ensure item is a tuple of chr, start, end
        if not ((isinstance(item, tuple) or isinstance(item, list)) and len(item) == 3):
            raise ValueError(
                "The item to check for must be a tuple of chr, start, end."
            )

        chr, start, end = item
        start = int(start)
        end = int(end)
        if chr not in self._trees:
            return False
        overlaps = self._trees[chr][start:end]
        return len(overlaps) > 0

    def __iter__(self):
        for tree in self._trees:
            yield self._trees[tree]


class Tokenizer:
    pass


class HardTokenizer(Tokenizer):
    """
    Tokenizer that computes overlaps between regions.
    """

    def __init__(
        self,
        regions: Union[List[str], str],
    ):
        """
        Create a new HardTokenizer.

        :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                        This can be either a list of regions or a path to a BED file containing regions.
        :param path_to_bedtools: The path to the bedtools executable.
        """
        self._regions = regions
        self._universe: Universe = Universe(validate_region_input(regions))

    @property
    def regions(self):
        return self._regions

    @property
    def universe(self):
        return self._universe

    def _tokenize_anndata(self, data: sc.AnnData) -> List[List[str]]:
        """
        Tokenize an anndata object into a list of lists of regions.

        :param data: The anndata object to tokenize.

        :return: A list of lists of regions.
        """
        data = convert_to_universe(data, self._universe)
        region_sets = anndata_to_regionsets(data)
        return region_sets

    def _tokenize_bed_file(self, data: str, fraction: float = 1e-9) -> List[str]:
        """
        Tokenize a BED file into a list of lists of regions.

        :param data: The path to the BED file to tokenize.

        :return: A list of lists of regions.
        """
        regions = validate_region_input(data)
        tokens = [
            "_".join([str(h) for h in hit]) for hit in self._universe.query(regions)
        ]
        return tokens

    def _tokenize_list(self, data: List[str]) -> List[List[str]]:
        """
        Tokenize a list of regions into a list of lists of regions.
        """
        regions = validate_region_input(data)
        hits = [h[0] for h in self._universe.query(regions)]
        tokens = ["_".join(hit) for hit in hits]
        return tokens

    def tokenize(
        self, data: Union[sc.AnnData, str, List[str]]
    ) -> Union[List[List[str]], List[str]]:
        """
        Tokenize a dataset. This will compute overlaps between regions and cells. Three
        types of data are accepted:
            1. Anndata object. This must have `chr`, `start`, and `end` values for the `.var` attribute.
            2. A path to a BED file containing regions.
            3. A list of regions.

        Regardless of the input, we return a list of lists of regions. Each list of regions
        corresponds to a cell.

        :param data: The data to tokenize.
        :raises ValueError: If the data is not one of the three accepted types.
        :return: A list of regions or a list of lists of regions (depending on the input type).
        """
        # check if data is anndata object
        if isinstance(data, sc.AnnData):
            return self._tokenize_anndata(data)
        # check if data is a path to a BED file
        elif isinstance(data, str):
            # ensure that the file exists
            if not os.path.exists(data):
                raise FileNotFoundError(f"Could not find file {data}.")
            return self._tokenize_bed_file(data)
        # check if data is a list of regions
        elif isinstance(data, list):
            return self._tokenize_list(data)
        # else, raise error
        else:
            raise ValueError(
                "Data must be one of the following types: anndata object, path to BED file, list of regions."
            )
