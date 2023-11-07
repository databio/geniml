import multiprocessing
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import scanpy as sc
from gtokenizers import (
    Region as GRegion,
    TreeTokenizer as GTreeTokenizer,
    TokenizedRegionSet as GTokenizedRegionSet,
    Universe as GUniverse,
)
from huggingface_hub import hf_hub_download
from rich.progress import track

from geniml.tokenization.split_file import split_file

from .const import UNIVERSE_FILE_NAME, CHR_KEY, START_KEY, END_KEY
from ..io import Region, RegionSet
from .hard_tokenization_batch import main as hard_tokenization
from .utils import time_str, Timer


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError


class ITTokenizer(Tokenizer):
    """
    A fast, in memory, tokenizer that uses `gtokenizers` - a rust based tokenizer. This
    tokenizer is the fastest tokenizer available. It is also the most memory efficient.
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

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Create a new tokenizer from a pretrained model's vocabulary.

        Usage:
        ```
        tokenizer = ITTokenizer.from_pretrained("path/to/universe.bed")
        ```

        :param str model_path: The path to the pretrained model on huggingface.
        """
        universe_file_path = hf_hub_download(model_path, UNIVERSE_FILE_NAME, **kwargs)
        return cls(universe_file_path)

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

    def _tokenize_anndata(self, adata: sc.AnnData) -> List[GTokenizedRegionSet]:
        """
        Tokenize an AnnData object. This is more involved, so it gets its own function.

        :param sc.AnnData query: The query to tokenize.
        """
        # extract regions from AnnData
        # its weird because of how numpy handle Intervals, the parent class of Region,
        # see here:
        # https://stackoverflow.com/a/43722306/13175187
        adata_features = [
            Region(chr, int(start), int(end))
            for chr, start, end in track(
                zip(adata.var[CHR_KEY], adata.var[START_KEY], adata.var[END_KEY]),
                total=adata.var.shape[0],
                description="Extracting regions from AnnData",
            )
        ]
        features = np.ndarray(len(adata_features), dtype=object)
        for i, region in enumerate(adata_features):
            features[i] = region
        del adata_features

        # tokenize
        tokenized = []
        for row in track(range(adata.shape[0]), total=adata.shape[0], description="Tokenizing"):
            _, non_zeros = adata.X[row].nonzero()
            regions = features[non_zeros]
            tokenized.append(self._tokenizer.tokenize(regions.tolist()))

        return tokenized

    def tokenize(self, query: Union[Region, RegionSet]) -> GTokenizedRegionSet:
        """
        Tokenize a Region or RegionSet into the universe

        :param Union[Region, RegionSet] query: The query to tokenize.
        """
        if isinstance(query, sc.AnnData):
            return self._tokenize_anndata(query)
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


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def hard_tokenization_main(
    src_folder: str,
    dst_folder: str,
    universe_file: str,
    fraction: float = 1e-9,
    file_list: List[str] = None,
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
        except Exception:
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
            _ = [r.get() for r in processes]
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
