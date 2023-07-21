import argparse
import glob
import json
import multiprocessing
import numpy as np
import os
import random
import requests
import shlex
import shutil
import subprocess
import sys
import yaml

from abc import ABC, abstractmethod
from queue import Queue

import gitk.region2vec.utils as utils
from gitk.tokenization.split_file import split_file
from .hard_tokenization_batch import main as hard_tokenization
from ..io import RegionSet, RegionSetCollection

from typing import List


# Should a tokenizer *hold* a universe, or take one as a parameter? Or both?


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError


class InMemTokenizer(Tokenizer):
    """Abstract class representing a tokenizer function"""

    @abstractmethod
    def tokenize(self, region_set: RegionSet, universe: RegionSet = None) -> RegionSet:
        """Tokenize a RegionSet"""
        raise NotImplementedError

    @abstractmethod
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
