import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pickle
from typing import Union, List, Tuple, Dict

import multiprocessing as mp
from multiprocessing.queues import Queue
import numpy as np

from .utils import genome_distance, load_genomic_embeddings


def get_topk_embed(
    i: int, K: int, embed: np.ndarray, dist: str = "cosine"
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the nearest K embedding indexes to the i-th embedding.

    Args:
        i (int): The index for the query embedding.
        K (int): The number of nearest embeddings to select.
        embed (np.ndarray): An array of embedding vectors
        dist (str, optional): The distance function used. Defaults to "cosine".

    Returns:
        tuple[np.ndarray, np.ndarray]: K indexes of nearest embeddings and the
            corresponding similarities.
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


def find_kneighbors(region_array: List[Tuple[str, int, int]], index: int, k: int) -> List[int]:
    """Finds the indexes of the K nearest regions of a query region on genome.

    region_array must be sorted, and all regions are on the same chromosome.

    Args:
        region_array (list[tuple[str, int, int]]): A list of (chromosome, start
            position, end position) tuples.
        index (int): The index of the query region.
        k (int): Specifies the number of nearest neighbors of the query region.

    Returns:
        list[int]: A list of indexes of the K nearest neighbors in
            region_array.
    """
    if len(region_array) < k:
        k = len(region_array)
    qregion = region_array[index]
    left_idx = max(index - k, 0)
    right_idx = min(index + k, len(region_array) - 1)
    rdist_arr = []
    for idx in range(left_idx, right_idx + 1):
        rdist_arr.append(genome_distance(qregion, region_array[idx]))
    rdist_arr = np.array(rdist_arr)
    Kneighbors_idx = np.argsort(rdist_arr)[1 : k + 1]
    Kneighbors_idx = Kneighbors_idx + left_idx
    return Kneighbors_idx


def calculate_overlap_bins(
    local_idx: int,
    K: int,
    chromo: str,
    region_array: List[Tuple[str, int, int]],
    region2index: dict[str, int],
    embed_rep: np.ndarray,
    res: int = 10,
    dist: str = "cosine",
    same_chromo: bool = True,
) -> np.ndarray:
    """Calculates the overlap ratios for a region.

    Calculates the overlap ratios for a region between its K-nearest neighor
    set obtained using genome distance and its K-nearest neighor set obtained
    using embedding distance. If res < K, then calculates ratios for size
    res*1, res*2, ..., min(res*n, K).

    Args:
        local_idx (int): The local index of a region on its chromosome.
        K (int): Specifies the number of nearest neighbors.
        chromo (str): Chromosome.
        region_array (list[tuple[str, int, int]]): A list of (chromosome, start
            position, end position) tuples.
        region2index (dict[str, int]): A dictionary of (region, index).
        embed_rep (np.ndarray): An array of embedding vectors.
        res (int, optional): Resolution. Size of neighborhood set. Defaults to
            10.
        dist (str, optional): Distance function. Defaults to "cosine".
        same_chromo (bool, optional): Whether to find nearest neighors on the
            same chromosome in the embedding space. Defaults to True.

    Returns:
        np.ndarray: An array of overlap ratios.
    """
    Kindices = find_kneighbors(region_array, local_idx, K)
    if len(Kindices) == 0:
        return 0
    str_kregions = [
        f"{chromo}:{region_array[k][0]}-{region_array[k][1]}" for k in Kindices
    ]  # sorted in ascending order
    _Krdist_global_indices = np.array([region2index[r] for r in str_kregions])

    if same_chromo:
        chr_regions = [
            f"{chromo}:{region_array[k][0]}-{region_array[k][1]}" for k in range(len(region_array))
        ]
        chr_global_indices = np.array([region2index[r] for r in chr_regions])
        chr_embeds = embed_rep[chr_global_indices]
        _Kedist_local_indices, _ = get_topk_embed(local_idx, K, chr_embeds, dist)
        _Kedist_global_indices = np.array([chr_global_indices[i] for i in _Kedist_local_indices])
    else:
        idx = region2index[f"{chromo}:{region_array[local_idx][0]}-{region_array[local_idx][1]}"]
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


def cal_snpr(ratio_embed: np.ndarray, ratio_random: np.ndarray) -> np.ndarray:
    """Calculates SNPR values.

    :param ratio_embed: Overlap ratios for query embeddings.
    :param ratio_random: Overlap ratios for random embeddings.

    :return: SNPR values.
    """
    res = np.log10((ratio_embed + 1.0e-10) / (ratio_random + 1.0e-10))
    res = np.maximum(res, 0)
    return res


def worker_func(
    i: int,
    K: int,
    chromo: str,
    region_array: List[Tuple[str, int, int]],
    embed_type: str,
    resolution: int,
    dist: str,
) -> np.ndarray:
    """Wrapper for calculate_overlap_bins

    Args:
        i (int): The local index of a region on its chromosome.
        K (int): Specifies the number of nearest neighbors.
        chromo (str): Chromosome.
        region_array (list[tuple[str, int, int]]): A list of (chromosome, start
            position, end position) tuples.
        embed_type (str): Embedding type, "region2vec" or "base".
        resolution (int): Resolution.
        dist (str): Distance function.

    Returns:
        np.ndarray: An array of overlap ratios.
    """
    var_dict = {}

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


def init_worker(
    embed_rep: np.ndarray, ref_embed: np.ndarray, region2index: Dict[str, int]
) -> None:
    """Initializes data used by workers.

    Args:
        embed_rep (np.ndarray): Query embeddings.
        ref_embed (np.ndarray): Random embeddings.
        region2index (dict[str, int]): A region to index dictionary.
    """
    var_dict = {}
    var_dict["embed_rep"] = embed_rep
    var_dict["ref_embed"] = ref_embed
    var_dict["region2vec_index"] = region2index


def get_npt_score(
    model_path: str,
    embed_type: str,
    K: int,
    num_samples: int = 100,
    seed: int = 0,
    resolution: int = 10,
    dist: str = "cosine",
    num_workers: int = 10,
) -> Dict[str, Union[int, np.ndarray, str]]:
    """Runs the NPT on a mdoel.

    If num_samples > 0, then randomly sample num_samples regions proportional
    from each chromosome. If num_samples == 0, all regions are used in the
    test. If K > resolution, then returns an array of NPT scores; otherwise,
    returns one NPT score.

    Args:
        model_path (str): The path to a model.
        embed_type (str):  The model type: "region2vec" or "base".
        K (int): Specifies the number of nearest neighbors.
        num_samples (int, optional): Number of embeddings used for evaluation.
            Defaults to 100.
        seed (int, optional): Random seed. Defaults to 0.
        resolution (int, optional): Resolution of a neighborhood set. Defaults
            to 10.
        dist (str, optional): Distance function. Defaults to "cosine".
        num_workers (int, optional): Number of parallel processes used.
            Defaults to 10.

    Returns:
        dict[str, Union[int, np.ndarray, str]]: NPT results in a dictionary
            "K": K,
            "Avg_qNPR": Average NPR ratios for query embeddings,
            "Avg_rNPR": Average NPR ratios for random embeddings,,
            "SNPR": SNPR values,
            "Resolution": Resolution,
            "Path": Model path,
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
                        len(region_array),
                        round(num_samples * chromo_ratios[chromo]),
                    )
                    indexes = np.random.permutation(len(region_array))[0:num]
                for i in indexes:
                    process_embed = pool.apply_async(
                        worker_func,
                        (i, K, chromo, region_array, "embed", resolution, dist),
                    )
                    process_random = pool.apply_async(
                        worker_func,
                        (
                            i,
                            K,
                            chromo,
                            region_array,
                            "random",
                            resolution,
                            dist,
                        ),
                    )
                    all_processes.append((process_embed, process_random))

            for i, (process_embed, process_random) in enumerate(all_processes):
                avg_ratio = (avg_ratio * count + process_embed.get()) / (count + 1)
                avg_ratio_ref = (avg_ratio_ref * count + process_random.get()) / (count + 1)
                count = count + 1
    else:
        for chromo in chromo_regions:
            region_array = chromo_regions[chromo]
            if num_samples == 0:  # exhaustive
                indexes = list(range(len(region_array)))
            else:
                num = min(
                    len(region_array),
                    round(num_samples * chromo_ratios[chromo]),
                )
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


def writer_multiprocessing(save_path: str, num: int, q: Queue) -> List[Tuple[str, float]]:
    """Writes results from multiple processes to a list.

    Args:
        save_path (str): The path to the saved results.
        num (int): The number of results.
        q (Queue): A multiprocessing queue.

    Returns:
        list[tuple[str, float]]: A list of (model path, NPT score) tuples.
    """
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
    batch: List[Tuple[str, str]],
    K: int,
    num_samples: int = 100,
    num_workers: int = 10,
    seed: int = 0,
    resolution: int = 10,
    dist: str = "cosine",
    save_path: str = None,
) -> List[Dict[str, Union[int, np.ndarray, str]]]:
    """Runs the NPT on a batch of models.

    Args:
        batch (list[tuple[str, str]]): A list of (model path, model type) tuples.
        K (int): Specifies the number of nearest neighbors.
        num_samples (int, optional): Number of embeddings used for evaluation.
            Defaults to 100.
        num_workers (int, optional): Number of parallel processes used.
            Defaults to 10.
        seed (int, optional): Random seed. Defaults to 0.
        resolution (int, optional): Resolution of a neighborhood set. Defaults
            to 10.
        dist (str, optional): Distance function. Defaults to "cosine".
        save_path (str, optional): Save the results to save_path. Defaults to
            None.

    Returns:
        list[dict[str, Union[int, np.ndarray, str]]]: A list of dictionaries of
            NPT results.
    """
    result_list = []
    for index, (path, embed_type) in enumerate(batch):
        result = get_npt_score(
            path,
            embed_type,
            K,
            num_samples,
            seed,
            resolution,
            dist,
            num_workers,
        )
        result_list.append(result)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result_list, f)
    return result_list


def npt_eval(
    batch: List[Tuple[str, str]],
    K: int,
    num_samples: int = 100,
    num_workers: int = 10,
    num_runs: int = 20,
    resolution: int = 10,
    dist: str = "cosine",
    save_folder: str = None,
) -> List[Tuple[str, np.ndarray, int]]:
    """Runs the NPT on a batch of models for multiple times.

    Args:
        batch (list[tuple[str, str]]): A list of (model path, model type) tuples.
        K (int): Specifies the number of nearest neighbors.
        num_samples (int, optional): Number of embeddings used for evaluation.
            Defaults to 100.
        num_workers (int, optional): Number of parallel processes used.
            Defaults to 10.
        num_runs (int, optional): Number of runs. Defaults to 20.
        resolution (int, optional): Resolution of a neighborhood set.
            Defaults to 10.
        dist (str, optional): Distance function. Defaults to "cosine".
        save_folder (str, optional): Folder to save the results from each run.
            Defaults to None.

    Returns:
        list[tuple[str, np.ndarray, int]]: A list of (model path, snprs from
            num_runs, resoultion) tuples.
    """
    results_seeds = []
    assert resolution <= K, "resolution <= K"
    for seed in range(num_runs):
        print(f"----------------Run {seed}----------------")
        save_path = os.path.join(save_folder, f"npt_eval_seed{seed}") if save_folder else None
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
