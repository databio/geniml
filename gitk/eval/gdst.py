import os
import pickle

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import glob
import multiprocessing as mp
import random
import time

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression

from .const import *
from .utils import (
    Timer,
    cosine_distance,
    genome_distance,
    load_genomic_embeddings,
)


def sample_from_vocab(vocab, num_samples, seed=42):
    chr_probs = {}
    region_dict = {}
    num_vocab = len(vocab)
    # build stat from vocab
    for region in vocab:
        chr_str, position = region.split(":")
        chr_str = chr_str.strip()
        start, end = position.split("-")
        start = int(start.strip())
        end = int(end.strip())
        chr_probs[chr_str] = chr_probs.get(chr_str, 0) + 1
        if chr_str in region_dict:
            region_dict[chr_str].append((start, end))
        else:
            region_dict[chr_str] = [(start, end)]
    total = sum([chr_probs[k] for k in chr_probs])
    chr_probs = [(k, chr_probs[k] / total) for k in chr_probs]

    count = 0

    chr_names = [t[0] for t in chr_probs]
    chr_probs = [t[1] for t in chr_probs]
    sampled_regions = []
    np.random.seed(seed)
    while count < num_samples:
        sel_chr = np.random.choice(chr_names, p=chr_probs)
        regions = region_dict[sel_chr]
        if len(regions) < 2:
            continue
        sel_indexes = np.random.choice(len(regions), 2, replace=False)
        r1, r2 = regions[sel_indexes[0]], regions[sel_indexes[1]]
        gdist = genome_distance(r1, r2)
        sampled_regions.append(
            (
                f"{sel_chr}:{r1[0]}-{r1[1]}",
                f"{sel_chr}:{r2[0]}-{r2[1]}",
                gdist,
            )
        )
        count += 1
    return sampled_regions


def remap_name(name):
    return name.split("/")[-3]


def convert_position(pos):
    if pos // 1e6 > 0:
        return f"{pos / 1e6:.4f} MB"
    elif pos // 1e3 > 0:
        return f"{pos / 1e3:.4f} KB"
    else:
        return f"{pos:.4f} B"


def get_gdst_results(save_paths):
    with open(save_paths[0], "rb") as f:
        results = pickle.load(f)
    num = len(results)
    gds_res = [[] for i in range(num)]
    models = ["" for i in range(num)]
    for path in save_paths:
        with open(path, "rb") as f:
            results = pickle.load(f)
            for i, res in enumerate(results):
                gds_res[i].append(res[1])
                models[i] = res[0]

    gds_arr = [(models[i], gds_res[i]) for i in range(num)]
    return gds_arr


def get_gdst_score(
    path,
    embed_type,
    num_samples=10000,
    seed=42,
    queue=None,
    worker_id=None,
):
    embed_rep, vocab = load_genomic_embeddings(path, embed_type)
    regions = sample_from_vocab(vocab, num_samples, seed)
    region2idx = {r: i for i, r in enumerate(vocab)}
    gdist_arr = [r[2] / GENOME_DIST_SCALAR for r in regions]
    edist_arr = np.array(
        [
            cosine_distance(embed_rep[region2idx[t[0]]], embed_rep[region2idx[t[1]]])
            for t in regions
        ]
    )
    gd_arr = list(zip(gdist_arr, edist_arr))
    X = np.array([[g[0]] for g in gd_arr])
    y = np.array([g[1] for g in gd_arr])
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    if queue:
        queue.put((worker_id, path, slope))
        return worker_id, path, slope
    else:
        return slope


def writer_multiprocessing(save_path, num, q):
    results = ["" for i in range(num)]
    while True:
        m = q.get()
        if m == "kill":
            break
        index = m[0]
        results[index] = (m[1], m[2])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    return results


def get_gdst_score_batch(batch, num_samples=10000, seed=42, save_path=None, num_workers=1):
    if num_workers <= 1:
        gds_arr = []
        for path, embed_type in batch:
            gds = get_gdst_score(path, embed_type, num_samples, seed)
            print(path, gds)
            gds_arr.append((path, gds))
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(gds_arr, f)
    else:
        manager = mp.Manager()
        queue = manager.Queue()
        with mp.Pool(processes=num_workers) as pool:
            writer = pool.apply_async(writer_multiprocessing, (save_path, len(batch), queue))
            all_processes = []
            for i, (path, embed_type) in enumerate(batch):
                process = pool.apply_async(
                    get_gdst_score,
                    (path, embed_type, num_samples, seed, queue, i),
                )
                all_processes.append(process)

            for process in all_processes:
                process.get()
            queue.put("kill")
            gds_arr = writer.get()
    return gds_arr


def gdst_eval(
    batch,
    num_runs=20,
    num_samples=1000,
    save_folder=None,
    num_workers=10,
):
    results_seeds = []
    for seed in range(num_runs):
        print(f"----------------Run {seed}----------------")
        save_path = os.path.join(save_folder, f"gdst_eval_seed{seed}") if save_folder else None
        result_list = get_gdst_score_batch(batch, num_samples, seed, save_path, num_workers)
        results_seeds.append(result_list)

    gds_res = [[] for i in range(len(batch))]
    for results in results_seeds:
        for i, res in enumerate(results):
            gds_res[i].append(res[1])
            assert res[0] == batch[i][0], "key == batch[i][0]"

    mean_gds = [np.array(r).mean() for r in gds_res]
    std_gds = [np.array(r).std() for r in gds_res]
    models = [t[0] for t in batch]
    for i in range(len(mean_gds)):
        print(f"{batch[i][0]}\n GDST score (std): {mean_gds[i]:.4f} ({std_gds[i]:.4f}) \n")
    gds_arr = [(batch[i][0], gds_res[i]) for i in range(len(batch))]
    return gds_arr
