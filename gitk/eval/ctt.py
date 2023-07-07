import os
import pickle
from typing import Union

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
from .utils import (
    Timer,
    cosine_distance,
    genome_distance,
    load_genomic_embeddings,
)


def explained_variance(bin_path, dim):
    bin_embed, _ = load_genomic_embeddings(bin_path, "base")
    pca_obj = PCA(n_components=dim).fit(bin_embed)
    ratio = pca_obj.explained_variance_ratio_.sum()
    return ratio


def get_ctt_score(
    path,
    embed_type,
    seed=42,
    num_data=10000,
    num_workers=10,
):
    """Implementation of hopkins' test. A score between 0 and 1, a score around 0.5 express a
    uniform distribution, a score around 0 indicate an evenely spaced distribution, and a score tending to 1 express a high cluster tendency.
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
        raise Exception(f"Number of samples ({num}) is too small")
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

    x = sum(sample_dist_to_nn**2)
    y = sum(random_dist_to_nn**2)

    if x + y == 0:
        raise Exception("The denominator of the hopkins statistics is zero")

    return y / (x + y)


def get_ctt_batch(batch, seed=42, num_data=10000, save_path=None, num_workers=10):
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
    batch,
    num_runs=20,
    num_data=10000,
    save_folder=None,
    num_workers=10,
):
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
