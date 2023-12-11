import os
import pickle

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import glob
import multiprocessing as mp
import random
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .const import *
from .utils import cosine_distance, genome_distance, load_genomic_embeddings


def get_ctt_score(
    path: str,
    embed_type: str,
    seed: int = 42,
    num_data: int = 10000,
    num_workers: int = 10,
) -> float:
    """Runs the cluster tendency test (CTT) on a model.

    Args:
        path (str): The path to a model.
        embed_type (str): The type of the model: "region2vec" or "base".
        seed (int, optional): Random seed. Defaults to 42.
        num_data (int, optional): Number of embeddings used for evaluation.
            Defaults to 10000.
        num_workers (int, optional): Number of parallel processes used.
            Defaults to 10.

    Raises:
        ValueError: The number of samples is too small.
        ZeroDivisionError: The denominator of the CTT score is zero.

    Returns:
        float: The CTT score for the model.
    """
    np.random.seed(seed)
    data, vocab = load_genomic_embeddings(path, embed_type)
    num_ori, dimension = data.shape
    if num_data < num_ori:
        num = num_data
    else:
        num = num_ori
    data = data[np.random.choice(num_ori, num)]
    if num < 100:
        raise ValueError(f"Number of samples ({num}) is too small")
    num_samples = int(num * CTT_TEST_RATIO)

    sel_indexes = np.random.choice(num, num_samples)
    data_sample = data[sel_indexes]
    neigh = NearestNeighbors(n_neighbors=2, n_jobs=num_workers).fit(data)
    sample_dist, _ = neigh.kneighbors(data_sample)
    sample_dist_to_nn = sample_dist[:, 1]

    max_vals = np.quantile(data, CTT_QUANTILE_MAX, axis=0)
    min_vals = np.quantile(data, CTT_QUANTILE_MIN, axis=0)
    random_points = np.random.uniform(min_vals, max_vals, (num_samples, dimension))

    random_dist, _ = neigh.kneighbors(random_points, n_neighbors=1)

    random_dist_to_nn = random_dist[:, 0]

    x = sum(sample_dist_to_nn ** 2)
    y = sum(random_dist_to_nn ** 2)

    if x + y == 0:
        raise ZeroDivisionError("The denominator is zero")

    return y / (x + y)


def get_ctt_batch(
    batch: list[tuple[str, str]],
    seed: int = 42,
    num_data: int = 10000,
    save_path: str = None,
    num_workers: int = 10,
) -> list[tuple[str, float]]:
    """Runs the cluster tendency test (CTT) on a batch of models.

    Args:
        batch (list[tuple[str, str]]): A list of (model path, model type)
            tuples. Model type could be "region2vec" or "base".
        seed (int, optional): Random seed. Defaults to 42.
        num_data (int, optional): Number of embeddings used for evaluation.
            Defaults to 10000.
        save_path (str, optional): Save the results to save_path. Defaults to
            None.
        num_workers (int, optional): Number of parallel processes used.
            Defaults to 10.

    Returns:
        list[tuple[str, float]]: A list of (model path, CTT score) tuples.
    """
    ctt_arr = []
    for path, embed_type in batch:
        ctt = get_ctt_score(path, embed_type, seed, num_data, num_workers)
        # print(f"{'/'.join(path.split('/')[-3:])}: {ctt:.4f}")
        ctt_arr.append((path, ctt))
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(ctt_arr, f)
    return ctt_arr


def ctt_eval(
    batch: list[tuple[str, str]],
    num_runs: int = 20,
    num_data: int = 10000,
    save_folder: str = None,
    num_workers: int = 10,
) -> list[tuple[str, list[float]]]:
    """Runs the CTT on a batch of models for multiple times.

    Runs the cluster tendency test (CTT) for a batch of models for num_runs
    times with different random seeds.


    Args:
        batch (list[tuple[str, str]]): A list of (model path, model type)
            tuples. Model type could be "region2vec" or "base".
        num_runs (int, optional): Number of runs. Defaults to 20.
        num_data (int, optional): Number of embeddings used for evaluation.
            Defaults to 10000.
        save_folder (str, optional): Folder to save the results from each run.
            Defaults to None.
        num_workers (int, optional): Number of parallel processes used.
            Defaults to 10.

    Returns:
        list[tuple[str, list[float]]]: A list of (model path, CTT scores from
            num_runs) tuples.
    """
    results_seeds = []
    for seed in range(num_runs):
        print(f"----------------Run {seed}----------------")
        save_path = os.path.join(save_folder, f"ctt_eval_seed{seed}") if save_folder else None
        result_list = get_ctt_batch(batch, seed, num_data, save_path, num_workers)
        results_seeds.append(result_list)

    ctt_res = [[] for i in range(len(batch))]
    for results in results_seeds:
        for i, res in enumerate(results):
            ctt_res[i].append(res[1])
            assert res[0] == batch[i][0], "key == batch[i][0]"

    mean_ctt = [np.array(r).mean() for r in ctt_res]
    std_ctt = [np.array(r).std() for r in ctt_res]
    models = [t[0] for t in batch]
    for i in range(len(mean_ctt)):
        print(f"{batch[i][0]}\n CTT (std): {mean_ctt[i]:.4f} ({std_ctt[i]:.4f}) \n")
    ctt_arr = [(batch[i][0], ctt_res[i]) for i in range(len(batch))]
    return ctt_arr
