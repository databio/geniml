import os
import pickle

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from matplotlib.lines import Line2D

from .utils import Timer, genome_distance, load_genomic_embeddings


def get_topk_embed(i, K, embed, dist="cosine"):
    """
    Return the indices for the most similar K regions to the i-th region
    embed is the embedding matrix for all the regions in the vocabulary of a region2vec model
    """
    num = len(embed)
    if dist == "cosine":
        nom = np.dot(embed[i : i + 1], embed.T)
        denom = np.linalg.norm(embed[i : i + 1]) * np.linalg.norm(embed, axis=1)
        sims = (nom / denom)[0]
        indexes = np.argsort(-sims)[1 : K + 1]
        s = sims[indexes]
    elif dist == "euclidean":
        dist = np.linalg.norm(embed[i : i + 1] - embed, axis=1)
        indexes = np.argsort(dist)[1 : K + 1]
        s = -dist[indexes]
    elif dist == "jaccard":
        nom = np.dot(embed[i : i + 1], embed.T)
        denom = ((embed[i : i + 1] + embed) > 0.0).sum(axis=1)
        sims = (nom / denom)[0]
        indexes = np.argsort(-sims)[1 : K + 1]
        s = sims[indexes]
    return indexes, s


def find_Kneighbors(region_array, index, K):
    """
    region_array must be sorted; all regions are on the same chromosome
    index is the index for the query region region_array[index]
    K is the number of nearest neighbors of the query region

    return: indices of the K nearest neighbors in region_array
    """
    if len(region_array) < K:
        K = len(region_array)
    qregion = region_array[index]
    left_idx = max(index - K, 0)
    right_idx = min(index + K, len(region_array) - 1)
    rdist_arr = []
    for idx in range(left_idx, right_idx + 1):
        rdist_arr.append(genome_distance(qregion, region_array[idx]))
    rdist_arr = np.array(rdist_arr)
    Kneighbors_idx = np.argsort(rdist_arr)[1 : K + 1]
    Kneighbors_idx = Kneighbors_idx + left_idx
    return Kneighbors_idx


def calculate_overlap_bins(
    local_idx,
    K,
    chromo,
    region_array,
    region2index,
    embed_rep,
    res=10,
    dist="cosine",
    same_chromo=True,
):
    Kindices = find_Kneighbors(region_array, local_idx, K)
    if len(Kindices) == 0:
        return 0
    str_kregions = [
        f"{chromo}:{region_array[k][0]}-{region_array[k][1]}" for k in Kindices
    ]  # sorted in ascending order
    _Krdist_global_indices = np.array([region2index[r] for r in str_kregions])

    if same_chromo:
        chr_regions = [
            f"{chromo}:{region_array[k][0]}-{region_array[k][1]}"
            for k in range(len(region_array))
        ]
        chr_global_indices = np.array([region2index[r] for r in chr_regions])
        chr_embeds = embed_rep[chr_global_indices]
        _Kedist_local_indices, _ = get_topk_embed(local_idx, K, chr_embeds, dist)
        _Kedist_global_indices = np.array(
            [chr_global_indices[i] for i in _Kedist_local_indices]
        )
    else:
        idx = region2index[
            f"{chromo}:{region_array[local_idx][0]}-{region_array[local_idx][1]}"
        ]
        _Kedist_global_indices, _ = get_topk_embed(idx, K, embed_rep, dist)

    bin_overlaps = []
    prev = 0
    assert res < K + 1, "resolution < K + 1"
    for i in range(res, K + 1, res):
        set1 = set(_Krdist_global_indices[prev:i])
        set2 = set(_Kedist_global_indices[prev:i])

        overlap = len(set1.intersection(set2)) / len(set1)
        bin_overlaps.append(overlap)

    return np.array(bin_overlaps)


def cal_snpr(ratio_embed, ratio_random):
    res = np.log10((ratio_embed + 1.0e-10) / (ratio_random + 1.0e-10))
    res = np.maximum(res, 0)
    return res


var_dict = {}


def worker_func(i, K, chromo, region_array, embed_type, resolution, dist):
    if embed_type == "embed":
        embeds = var_dict["embed_rep"]
    elif embed_type == "random":
        embeds = var_dict["ref_embed"]
    nprs = calculate_overlap_bins(
        i,
        K,
        chromo,
        region_array,
        var_dict["region2vec_index"],
        embeds,
        resolution,
        dist,
    )
    return nprs


def init_worker(embed_rep, ref_embed, region2index):
    var_dict["embed_rep"] = embed_rep
    var_dict["ref_embed"] = ref_embed
    var_dict["region2vec_index"] = region2index


def get_npt_score(
    model_path,
    embed_type,
    K,
    num_samples=100,
    seed=0,
    resolution=10,
    dist="cosine",
    num_workers=10,
):
    """
    If sampling > 0, then randomly sample num_samples regions in total (proportional for each chromosome)

    If num_samples == 0, all regions are used in calculation
    """
    embed_rep, regions_r2v = load_genomic_embeddings(model_path, embed_type)

    region2index = {r: i for i, r in enumerate(regions_r2v)}
    # Group regions by chromosomes
    chromo_regions = {}
    for v in regions_r2v:
        chromo, region = v.split(":")  # e.g. chr1:100-1000
        chromo = chromo.strip()  # remove possible spaces
        region = region.strip()  # remove possible spaces
        start, end = region.split("-")
        start = int(start.strip())
        end = int(end.strip())
        if chromo not in chromo_regions:
            chromo_regions[chromo] = [(start, end)]
        else:
            chromo_regions[chromo].append((start, end))

    # sort regions in each chromosome
    chromo_ratios = {}
    for chromo in chromo_regions:
        region_array = chromo_regions[chromo]
        chromo_regions[chromo] = sorted(region_array, key=lambda x: x[0])
        chromo_ratios[chromo] = len(region_array) / len(regions_r2v)

    num_regions, num_dim = embed_rep.shape

    np.random.seed(seed)

    ref_embed = (np.random.rand(num_regions, num_dim) - 0.5) / num_dim

    avg_ratio = 0.0
    avg_ratio_ref = 0.0
    count = 0

    if num_workers > 1:
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(embed_rep, ref_embed, region2index),
        ) as pool:
            all_processes = []
            for chromo in chromo_regions:
                region_array = chromo_regions[chromo]
                if num_samples == 0:  # exhaustive
                    indexes = list(range(len(region_array)))
                else:
                    num = min(
                        len(region_array), round(num_samples * chromo_ratios[chromo])
                    )
                    indexes = np.random.permutation(len(region_array))[0:num]
                for i in indexes:
                    process_embed = pool.apply_async(
                        worker_func,
                        (i, K, chromo, region_array, "embed", resolution, dist),
                    )
                    process_random = pool.apply_async(
                        worker_func,
                        (i, K, chromo, region_array, "random", resolution, dist),
                    )
                    all_processes.append((process_embed, process_random))

            for i, (process_embed, process_random) in enumerate(all_processes):
                avg_ratio = (avg_ratio * count + process_embed.get()) / (count + 1)
                avg_ratio_ref = (avg_ratio_ref * count + process_random.get()) / (
                    count + 1
                )
                count = count + 1
    else:
        for chromo in chromo_regions:
            region_array = chromo_regions[chromo]
            if num_samples == 0:  # exhaustive
                indexes = list(range(len(region_array)))
            else:
                num = min(len(region_array), round(num_samples * chromo_ratios[chromo]))
                indexes = np.random.permutation(len(region_array))[0:num]
            for i in indexes:
                nprs_embed = calculate_overlap_bins(
                    i,
                    K,
                    chromo,
                    region_array,
                    region2index,
                    embed_rep,
                    resolution,
                    dist,
                )
                nprs_random = calculate_overlap_bins(
                    i,
                    K,
                    chromo,
                    region_array,
                    region2index,
                    ref_embed,
                    resolution,
                    dist,
                )
                avg_ratio = (avg_ratio * count + nprs_embed) / (count + 1)
                avg_ratio_ref = (avg_ratio_ref * count + nprs_random) / (count + 1)
                count = count + 1
    snprs = cal_snpr(avg_ratio, avg_ratio_ref)

    ratio_msg = " ".join([f"{r:.6f}" for r in avg_ratio])
    ratio_ref_msg = " ".join([f"{r:.6f}" for r in avg_ratio_ref])
    snprs_msg = " ".join([f"{r:.6f}" for r in snprs])
    result = {
        "K": K,
        "Avg_qNPR": avg_ratio,
        "Avg_rNPR": avg_ratio_ref,
        "SNPR": snprs,
        "Resolution": resolution,
        "Path": model_path,
    }
    return result


def writer_multiprocessing(save_path, num, q):
    results = [[] for i in range(num)]
    while True:
        m = q.get()
        if m == "kill":
            break
        worker_id = m[0]
        results[worker_id] = m[1]
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    return results


def get_npt_score_batch(
    batch,
    K,
    num_samples=100,
    num_workers=10,
    seed=0,
    resolution=10,
    dist="cosine",
    save_path=None,
):
    result_list = []
    for index, (path, embed_type) in enumerate(batch):
        result = get_npt_score(
            path, embed_type, K, num_samples, seed, resolution, dist, num_workers
        )
        result_list.append(result)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result_list, f)
    return result_list


def npt_eval(
    batch,
    K,
    num_samples=100,
    num_workers=10,
    num_runs=20,
    resolution=10,
    dist="cosine",
    save_folder=None,
):
    results_seeds = []
    assert resolution <= K, "resolution <= K"
    for seed in range(num_runs):
        print(f"----------------Run {seed}----------------")
        save_path = (
            os.path.join(save_folder, f"npt_eval_seed{seed}") if save_folder else None
        )
        result_list = get_npt_score_batch(
            batch,
            K,
            num_samples=num_samples,
            num_workers=num_workers,
            seed=seed,
            resolution=resolution,
            dist=dist,
            save_path=save_path,
        )
        results_seeds.append(result_list)
    snpr_results = [[] for i in range(len(batch))]
    paths = ["" for i in range(len(batch))]
    for results in results_seeds:
        for i, result in enumerate(results):
            key = result["Path"]
            snpr_results[i].append(result["SNPR"])
            paths[i] = key
    snpr_results = [np.array(v) for v in snpr_results]
    for i in range(len(batch)):
        snpr_arr = snpr_results[i]
        avg_snprs = snpr_arr.mean(axis=0)
        std_snprs = snpr_arr.std(axis=0)
        msg = " ".join([f"{m:.4f}({s:.4f})" for m, s in zip(avg_snprs, std_snprs)])
        print(f"{paths[i]}\nSNPRs:{msg}\n")
    snpr_results = [(paths[i], snpr_results[i], resolution) for i in range(len(batch))]
    return snpr_results


def get_npt_results(save_paths):
    snpr_results = {}
    for save_path in save_paths:
        with open(save_path, "rb") as f:
            results = pickle.load(f)
            for result in results:
                key = result["Path"]
                resolution = result["Resolution"]
                if key in snpr_results:
                    snpr_results[key].append(result["SNPR"])
                else:
                    snpr_results[key] = [result["SNPR"]]
    snpr_results = [(k, np.array(v), resolution) for k, v in snpr_results.items()]
    return snpr_results


def snpr_plot(snpr_data, row_labels=None, legend_pos=(0.25, 0.6), filename=None):
    snpr_vals = [
        (k, v.sum(axis=1).mean(), v.sum(axis=1).std()) for k, v, res in snpr_data
    ]
    cmap = plt.get_cmap("Set1")
    cmaplist = [cmap(i) for i in range(9)]
    if row_labels is None:
        row_labels = [k for k, v, s in snpr_vals]
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_snpr_tuple = [(i, r[1]) for i, r in enumerate(snpr_vals)]
    mean_snpr_tuple = sorted(mean_snpr_tuple, key=lambda x: -x[1])
    mean_snpr = [t[1] for t in mean_snpr_tuple]
    indexes = [t[0] for t in mean_snpr_tuple]
    std_snpr = [snpr_vals[i][2] for i in indexes]
    row_labels = [row_labels[i] for i in indexes]
    ax.set_xticks(list(range(1, len(mean_snpr) + 1)))
    ax.set_xticklabels(row_labels)
    ax.errorbar(
        range(1, len(mean_snpr) + 1),
        mean_snpr,
        yerr=std_snpr,
        fmt="o",
        ms=10,
        mfc=cmaplist[1],
        mec=cmaplist[8],
        ecolor=cmaplist[2],
        elinewidth=3,
        capsize=5,
    )
    ax.set_ylabel("SNPR")
    _ = plt.setp(
        ax.get_xticklabels(), rotation=-15, ha="left", va="top", rotation_mode="anchor"
    )
    patches = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=cmaplist[1],
            markersize=12,
            mec=cmaplist[8],
        ),
        Line2D([0], [0], color=cmaplist[2], lw=4),
    ]
    legend = ax.legend(
        labels=["SNPR", "SNPR standard deviation"],
        handles=patches,
        bbox_to_anchor=legend_pos,
        loc="center left",
        borderaxespad=0,
        fontsize=12,
        frameon=True,
    )
    ax.grid("on")
    if filename:
        fig.savefig(filename, bbox_inches="tight")
