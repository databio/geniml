from typing import List, Union

import scanpy as sc
import os

from .const import DEFAULT_PATH_TO_BEDTOOLS, UNIVERSE_FILE_NAME
from .utils import (
    make_cache_dir,
    get_cache_dir,
    convert_to_universe,
    anndata_to_regionsets,
)


class Tokenizer:
    pass


class HardTokenizer(Tokenizer):
    """
    Tokenizer that computes overlaps between regions.
    """

    def __init__(
        self,
        regions: Union[List[str], str],
        path_to_bedtools: str = DEFAULT_PATH_TO_BEDTOOLS,
    ):
        """
        Create a new HardTokenizer.

        :param regions: The regions to use for tokenization. This can be thought of as a vocabulary.
                        This can be either a list of regions or a path to a BED file containing regions.
        :param path_to_bedtools: The path to the bedtools executable.
        """
        self.regions = regions
        self.path_to_bedtools = path_to_bedtools

        # write regions to file in cache dir
        self.__cache_dir = get_cache_dir()
        self.__regions_file = os.path.join(self.__cache_dir, UNIVERSE_FILE_NAME)
        self.__write_regions_to_file()

        make_cache_dir()

    def __write_regions_to_file(self):
        """
        Write regions to file in cache dir. This can take one of two forms:
        1. A list of regions.
        2. A path to a BED file containing regions.

        If the regions are a list, they are written to a file in the cache dir.
        If the regions are a path to a BED file, the file is copied to the cache dir.

        :raises FileNotFoundError: If the path to the BED file does not exist.
        """
        # check for path to file
        if isinstance(self.regions, str):
            # check if file exists
            if not os.path.exists(self.regions):
                raise FileNotFoundError(
                    f"Could not find file {self.regions} containing regions."
                )
            # copy file to cache dir
            os.system(f"cp {self.regions} {self.__regions_file}")

        # else, we are going to write the regions to a file
        # verify that the regions are a list, should be in the form chr_start_end
        elif isinstance(self.regions, list):
            if not all(
                [
                    isinstance(region, str) and len(region.split("_")) == 3
                    for region in self.regions
                ]
            ):
                raise ValueError(
                    "Regions must be a list of strings in the form chr_start_end."
                )
            # write regions to file
            with open(self.__regions_file, "w") as f:
                # split each region into chr, start, end
                for region in self.regions:
                    chr, start, end = region.split("_")
                    f.write(f"{chr}\t{start}\t{end}\n")

    def __tokenize_anndata(self, data: sc.AnnData) -> List[List[str]]:
        """
        Tokenize an anndata object into a list of lists of regions.

        :param data: The anndata object to tokenize.

        :return: A list of lists of regions.
        """
        data = convert_to_universe(data, self.__regions_file)
        region_sets = anndata_to_regionsets(data)
        return region_sets

    def __tokenize_bed_file(self, data: str, fraction: float = 1e-9) -> List[str]:
        """
        Tokenize a BED file into a list of lists of regions.

        :param data: The path to the BED file to tokenize.

        :return: A list of lists of regions.
        """
        # perform overlap analysis with bedtools
        os.system(
            f"{self.path_to_bedtools} intersect -a {data} -b {self.__regions_file} -wa -wb > {self.__cache_dir}/overlaps.bed"
        )
        # read in overlaps
        overlaps = []
        with open(f"{self.__cache_dir}/overlaps.bed", "r") as f:
            for line in f:
                line = line.strip().split("\t")
                # join chr, start, end with _
                region = "_".join(line[0:3])
                overlaps.append(region)

        return overlaps

    def __tokenize_list(self, data: List[str]) -> List[List[str]]:
        """
        Tokenize a list of regions into a list of lists of regions.
        """
        # write to bed file, and perform __tokenize_bed_file
        with open(f"{self.__cache_dir}/regions.bed", "w") as f:
            for region in data:
                # split region into chr, start, end
                chr, start, end = region.split("_")
                f.write(f"{chr}\t{start}\t{end}\n")

        return self.__tokenize_bed_file(f"{self.__cache_dir}/regions.bed")

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
            return self.__tokenize_anndata(data)
        # check if data is a path to a BED file
        elif isinstance(data, str):
            # ensure that the file exists
            if not os.path.exists(data):
                raise FileNotFoundError(f"Could not find file {data}.")
            return self.__tokenize_bed_file(data)
        # check if data is a list of regions
        elif isinstance(data, list):
            return self.__tokenize_list(data)
        # else, raise error
        else:
            raise ValueError(
                "Data must be one of the following types: anndata object, path to BED file, list of regions."
            )
