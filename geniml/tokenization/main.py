import multiprocessing
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import scanpy as sc
from huggingface_hub import hf_hub_download
from rich.progress import track

from geniml.tokenization.split_file import split_file
from geniml.io import Region, RegionSet
from genimtools.tokenizers import (
    TreeTokenizer as GTreeTokenizer,
    Region as GRegion,
)

from .hard_tokenization_batch import main as hard_tokenization
from .utils import Timer, time_str


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TreeTokenizer(Tokenizer):
    """
    A fast, in memory, tokenizer that uses `gtokenizers` - a rust based tokenizer.

    This should be used to tokenize bulk data, like BED files. If you need to tokenize single cell data, use
    the `AnnDataTokenizer` instead.
    """

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Create a new tokenizer from a pretrained model's vocabulary.

        Usage:
        ```
        tokenizer = TreeTokenizer.from_pretrained("path/to/universe.bed")
        ```

        :param str model_path: The path to the pretrained model on huggingface.
        """
        universe_file_path = hf_hub_download(model_path, "universe.bed")
        return cls(universe_file_path, **kwargs)

    def __init__(self, universe: str) -> GTreeTokenizer:
        """
        Create a new tokenizer.

        This tokenizer only accepts a path to a BED file containing regions.

        :param str universe: The universe to use for tokenization.
        """
        self._tokenizer = GTreeTokenizer(universe)

    @property
    def universe(self):
        return self._tokenizer.universe

    def tokenize(self, query: Union[str, RegionSet]) -> List[List[Region]]:
        """
        Tokenize a Region or RegionSet into the universe

        :param Union[Region, RegionSet] query: The query to tokenize.
        """
        if isinstance(query, sc.AnnData) or isinstance(query, RegionSet):
            result = self._tokenizer(query)
            return result.to_regions()
        else:
            raise ValueError(
                f"Please pass a RegionSet object or a path to a BED file. You passed: {type(query)}"
            )

    def encode(self, query: sc.AnnData) -> List[List[int]]:
        """
        Tokenize an AnnData object to IDs.

        :param sc.AnnData query: The query to tokenize.
        """
        if isinstance(query, sc.AnnData) or isinstance(query, RegionSet):
            result = self._tokenizer(query)
            return result.to_ids()
        else:
            raise ValueError(
                f"Please pass a RegionSet object or a path to a BED file. You passed: {type(query)}"
            )

    def decode(self, query: List[List[int]]) -> List[List[Region]]:
        """
        Decode a list of IDs back to regions.

        :param List[List[int]] query: The query to decode.
        """
        return [self._tokenizer.decode(ids) for ids in query]

    def padding_token(self) -> GRegion:
        return self._tokenizer.padding_token

    def padding_token_id(self) -> int:
        return self._tokenizer.padding_token_id

    def unknown_token(self) -> GRegion:
        return self._tokenizer.unknown_token

    def unknown_token_id(self) -> int:
        return self._tokenizer.unknown_token_id

    def mask_token(self) -> GRegion:
        return self._tokenizer.mask_token

    def mask_token_id(self) -> int:
        return self._tokenizer.mask_token_id

    def cls_token(self) -> GRegion:
        return self._tokenizer.cls_token

    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    def bos_token(self) -> GRegion:
        return self._tokenizer.bos_token

    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id

    def eos_token(self) -> GRegion:
        return self._tokenizer.eos_token

    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    def sep_token(self) -> GRegion:
        return self._tokenizer.sep_token

    def sep_token_id(self) -> int:
        return self._tokenizer.sep_token_id

    def __len__(self):
        return len(self.universe.regions)

    def __call__(self, query: Union[str, RegionSet]) -> List[List[Region]]:
        if isinstance(query, str) or isinstance(query, RegionSet):
            result = self._tokenizer(query)
            return result
        else:
            raise NotImplementedError("Only AnnData is supported for this tokenizer.")

    def __repr__(self):
        return "TreeTokenizer()"

    def __str__(self):
        return "TreeTokenizer()"


class AnnDataTokenizer(Tokenizer):
    """
    A fast, in memory, tokenizer that uses `gtokenizers` - a rust based tokenizer.

    This is actually a wrapper around the core TreeTokenizer. Ideally, we'd tokenize the
    AnnData natively in Rust, but that's a bit more involved. So, we'll just use the
    Python wrapper for now.
    """

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Create a new tokenizer from a pretrained model's vocabulary.

        Usage:
        ```
        tokenizer = AnnDataTokenizer.from_pretrained("path/to/universe.bed")
        ```

        :param str model_path: The path to the pretrained model on huggingface.
        """
        universe_file_path = hf_hub_download(model_path, "universe.bed")
        return cls(universe_file_path, **kwargs)

    def __init__(self, universe: str = None, verbose: bool = False):
        """
        Create a new tokenizer.

        This tokenizer only accepts a path to a BED file containing regions.

        :param str universe: The universe to use for tokenization.
        """
        self.verbose = verbose
        if universe is not None:
            self._tokenizer: GTreeTokenizer = GTreeTokenizer(universe)
        else:
            self._tokenizer = None

    def _tokenize_anndata(self, adata: sc.AnnData) -> List[List[Region]]:
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
                zip(adata.var["chr"], adata.var["start"], adata.var["end"]),
                total=adata.var.shape[0],
                description="Extracting regions from AnnData",
                disable=not self.verbose,
            )
        ]
        features = np.ndarray(len(adata_features), dtype=object)
        for i, region in enumerate(adata_features):
            features[i] = region
        del adata_features

        # tokenize
        tokenized = []
        for row in track(
            range(adata.shape[0]),
            total=adata.shape[0],
            description="Tokenizing",
            disable=not self.verbose,
        ):
            start = time.time()
            _, non_zeros = adata.X[row].nonzero()
            end = time.time()

            print(f"Nonzero time: {end - start}")

            start = time.time()
            regions = features[non_zeros]
            end = time.time()

            print(f"Regions time: {end - start}")

            start = time.time()
            tokenized.append(self._tokenizer(regions))
            end = time.time()

            print(f"Tokenize time: {end - start}")

        return tokenized

    @property
    def universe(self):
        return self._tokenizer.universe

    def tokenize(self, query: sc.AnnData) -> List[List[Region]]:
        """
        Tokenize a Region or RegionSet into the universe

        :param Union[Region, RegionSet, sc.AnnData] query: The query to tokenize.
        :param bool ids_only: Whether to return only the IDs or the full TokenizedRegionSet
        :param bool as_strings: Whether to return the IDs as strings or ints
        """
        if isinstance(query, sc.AnnData):
            result = self._tokenize_anndata(query)
            return [result.to_regions() for result in result]
        elif isinstance(query, str):
            query = sc.read_h5ad(query)
            result = self._tokenize_anndata(query)
            return [result.to_regions() for result in result]
        else:
            raise NotImplementedError("Only AnnData is supported right now.")

    def encode(self, query: sc.AnnData) -> List[List[int]]:
        """
        Tokenize an AnnData object to IDs.

        :param sc.AnnData query: The query to tokenize.
        """
        if isinstance(query, sc.AnnData):
            result = self._tokenize_anndata(query)
            return [t.to_ids() for t in result]
        elif isinstance(query, str):
            query = sc.read_h5ad(query)
            result = self._tokenize_anndata(query)
            return [t.to_ids() for t in result]
        else:
            raise NotImplementedError("Only AnnData is supported right now.")

    def decode(self, query: List[List[int]]) -> List[List[Region]]:
        """
        Decode a list of IDs back to regions.

        :param List[List[int]] query: The query to decode.
        """
        return [self._tokenizer.decode(ids) for ids in query]

    def padding_token(self) -> GRegion:
        return self._tokenizer.padding_token

    def padding_token_id(self) -> int:
        return self._tokenizer.padding_token_id

    def unknown_token(self) -> GRegion:
        return self._tokenizer.unknown_token

    def unknown_token_id(self) -> int:
        return self._tokenizer.unknown_token_id

    def mask_token(self) -> GRegion:
        return self._tokenizer.mask_token

    def mask_token_id(self) -> int:
        return self._tokenizer.mask_token_id

    def cls_token(self) -> GRegion:
        return self._tokenizer.cls_token

    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    def bos_token(self) -> GRegion:
        return self._tokenizer.bos_token

    def bos_token_id(self) -> int:
        return self._tokenizer.bos_token_id

    def eos_token(self) -> GRegion:
        return self._tokenizer.eos_token

    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    def sep_token(self) -> GRegion:
        return self._tokenizer.sep_token

    def sep_token_id(self) -> int:
        return self._tokenizer.sep_token_id

    def __len__(self):
        return len(self.universe.regions)

    def __call__(self, query: sc.AnnData) -> List[List[Region]]:
        if isinstance(query, sc.AnnData):
            result = self._tokenize_anndata(query)
            return result
        elif isinstance(query, str):
            query = sc.read_h5ad(query)
            result = self._tokenize_anndata(query)
            return result
        else:
            raise NotImplementedError("Only AnnData is supported for this tokenizer.")

    def __repr__(self):
        return "AnnDataTokenizer()"

    def __str__(self):
        return "AnnDataTokenizer()"


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
        dst_folder (str): The folder to store tokenized BED files.
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
