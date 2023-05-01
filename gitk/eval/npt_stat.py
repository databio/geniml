import pickle
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import time
import argparse
from gensim.models import Word2Vec
import time
import multiprocessing as mp
from gitk.eval import load_genomic_embeddings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import KDTree

class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return "{:.1f}h".format(x / 3600)
        if x >= 60:
            return "{}m".format(round(x / 60))
        return "{}s".format(x)

var_dict = {}


def worker_func(regions, num_tests, num_neighbors):
    rep_r2i = var_dict['rep_r2i']
    bin_r2i = var_dict['bin_r2i']
    sub_rep_embed = np.vstack([var_dict['rep_embed'][rep_r2i[r]] for r in regions])
    sub_bin_embed = np.vstack([var_dict['bin_embed'][bin_r2i[r]] for r in regions])
    avg_npr = exact_npt(sub_rep_embed, sub_bin_embed, num_tests, num_neighbors)
    return avg_npr


def init_worker(rep_embed, bin_embed, rep_r2i, bin_r2i):
    var_dict["rep_embed"] = rep_embed
    var_dict["bin_embed"] = bin_embed
    var_dict["rep_r2i"] = rep_r2i
    var_dict["bin_r2i"] = bin_r2i

def get_topk_embed(query, embed, K, dist="cosine"):
    neighbors = []
    if dist == "cosine":
        embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
        sims = np.dot(embed, embed.T)
        for q in query:
            indexes = np.argsort(-sims[q])[1:K+1]
            neighbors.append(indexes)
   
    return neighbors

def exact_npt(rep_embed, bin_embed, num_tests, num_neighbors):
    num_samples = len(rep_embed)
    assert num_samples >= num_tests, "num_samples ({}) >= num_tests ({}) ".format(num_samples, num_tests)
    assert num_samples >= num_neighbors+1, "num_samples ({}) >= num_neighbors+1 ({}) ".format(num_samples, num_neighbors)
    assert num_neighbors > 0, "num_neighbors > 0"
    sel_indexes = np.random.permutation(num_samples)[0:num_tests]
    
    rep_neighbors = get_topk_embed(sel_indexes, rep_embed, num_neighbors)
    bin_neighbors = get_topk_embed(sel_indexes, bin_embed, num_neighbors)
    
    # rep_tree = KDTree(rep_embed)
    # bin_tree = KDTree(bin_embed)
    # rep_query = rep_embed[sel_indexes]
    # bin_query = bin_embed[sel_indexes]
    # _, rep_neighbors = rep_tree.query(rep_query, k=num_neighbors)
    # _, bin_neighbors = bin_tree.query(bin_query, k=num_neighbors)

    ratio_arr = []
    for i in range(num_tests):
        rep_neighbor_set = set(rep_neighbors[i])
        bin_neighbor_set = set(bin_neighbors[i])
        num_overlaps = len(bin_neighbor_set.intersection(rep_neighbor_set))
        overlap_ratio = num_overlaps / num_neighbors
        ratio_arr.append(overlap_ratio)
    ratio_arr = np.array(ratio_arr)
    return ratio_arr.mean()
    
def npt(
    model_path, embed_type, bin_path, num_tests=10000, num_neighbors=20, group_size=1000, test_per_group=100, num_workers=10, seed=0
):
    """
    Sample a group of regions, then run the exact neighborhood preserving test therein
    """
    timer = Timer()
    np.random.seed(seed)
    rep_embed, rep_vocab = load_genomic_embeddings(model_path, embed_type)
    rep_r2i = {r: i for i, r in enumerate(rep_vocab)}

    bin_embed, bin_vocab = load_genomic_embeddings(bin_path, 'base')
    bin_r2i = {r: i for i, r in enumerate(bin_vocab)}
    
    num_regions, num_dim = rep_embed.shape

    
    num_groups = num_tests // test_per_group
    last_group_tests = num_tests - test_per_group * num_groups
    if last_group_tests == 0:
        test_per_groups = [test_per_group] * num_groups
    else:
        test_per_groups = [test_per_group] * num_groups + [last_group_tests]
        num_groups += 1
    
    
    avg_ratio = 0.0
    count = 0

    if num_workers > 1:
        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(rep_embed, bin_embed, rep_r2i, bin_r2i),
        ) as pool:
            all_processes = []
            for g in range(num_groups):
                sel_indexes = np.random.permutation(num_regions)[0:group_size]
                regions = [rep_vocab[i] for i in sel_indexes]
                process = pool.apply_async(
                    worker_func, (regions, test_per_groups[g], num_neighbors)
                )
                all_processes.append(process)

            for i, process in enumerate(all_processes):
                avg_ratio = (avg_ratio * count + process.get()*test_per_groups[i]) / (count + test_per_groups[i])
                count = count + test_per_groups[i]
    else:
        for g in range(num_groups):
            sel_indexes = np.random.permutation(num_regions)[0:group_size]
            regions = [rep_vocab[i] for i in sel_indexes]
            sub_rep_embed = np.vstack([rep_embed[rep_r2i[r]] for r in regions])
            sub_bin_embed = np.vstack([bin_embed[bin_r2i[r]] for r in regions])
            avg_npr = exact_npt(sub_rep_embed, sub_bin_embed, test_per_groups[g], num_neighbors)
        
            avg_ratio = (avg_ratio * count + avg_npr*test_per_groups[g]) / (count + test_per_groups[g])
            count = count + test_per_groups[g]
    
    print(model_path)

    print(
        "[seed={}] K={}: {:.6f} ({} samples)\n".format(
            seed, num_neighbors, avg_ratio, num_tests
        )
    )
    elapsed_time = timer.measure()
    print("Elapsed time:", elapsed_time)
    return (model_path, avg_ratio)




def npt_batch(
    batch, bin_path, num_tests=10000, num_neighbors=20, group_size=1000, test_per_group=100, num_workers=10, seed=0, save_path=None
):
    print("Total number of models: {}".format(len(batch)))
    result_list = []
    for index, (path, embed_type) in enumerate(batch):
        result = npt(
            path, embed_type, bin_path, num_tests, num_neighbors, group_size, test_per_group, num_workers, seed
        )
        result_list.append(result)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result_list, f)
    return result_list


def npt_eval(batch, bin_path, num_runs=20, num_tests=10000, num_neighbors=20, group_size=1000, test_per_group=100, num_workers=10, save_folder=None):
    results_seeds = []
    for seed in range(num_runs):
        print("----------------Run {}----------------".format(seed))
        save_path = (
            os.path.join(save_folder, "npt_eval_seed{}".format(seed))
            if save_folder
            else None
        )
        result_list = npt_batch(
            batch,
            bin_path,
            num_tests=num_tests,
            num_neighbors=num_neighbors,
            group_size=group_size,
            test_per_group=test_per_group,
            num_workers=num_workers, 
            seed=seed,
            save_path=save_path,
        )
        results_seeds.append(result_list)
    npr_results = [[] for i in range(len(batch))]
    paths = ["" for i in range(len(batch))]
    for results in results_seeds:
        for i,result in enumerate(results):
            paths[i] = result[0]
            npr_results[i].append(result[1])
    npr_results = [np.array(v) for v in npr_results]
    print(npr_results[0].shape)
    for i in range(len(batch)):
        print(
            "{}\n AvgNPR (std):{:.6f} ({:.6f})".format(
                paths[i], npr_results[i].mean(), npr_results[i].std()
            )
        )
    npr_results = [(paths[i], npr_results[i]) for i in range(len(batch))]
    return npr_results


def get_npt_results(save_paths):
    npr_results = {}
    for save_path in save_paths:
        with open(save_path, "rb") as f:
            results = pickle.load(f)
            for result in results:
                key = result[0]
                if key in npr_results:
                    npr_results[key].append(result[1])
                else:
                    npr_results[key] = [result[1]]
    npr_results = [(k, np.array(v)) for k, v in npr_results.items()]
    return npr_results


def get_avg_npr(npr_data, row_labels=None):
    npr_vals = [(k, v.mean(), v.std()) for k, v in npr_data]
    if row_labels is None:
        row_labels = [k for k, v, s in npr_vals]
    mean_npr = [r[1] for r in npr_vals]
    std_npr = [r[2] for r in npr_vals]
    sem_npr = [g/np.sqrt(len(npr_data[0][1])) for g in std_npr]
    return mean_npr, sem_npr, row_labels


def npr_plot(npr_data, row_labels=None, legend_pos=(0.25, 0.6), filename=None):
    mean_npr, sem_npr, row_labels = get_avg_npr(npr_data, row_labels)
    cmap = plt.get_cmap("Set1")
    cmaplist = [cmap(i) for i in range(9)]
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_npr_tuple = [(i, r) for i, r in enumerate(mean_npr)]
    mean_npr_tuple = sorted(mean_npr_tuple, key=lambda x: -x[1])
    mean_npr = [t[1] for t in mean_npr_tuple]
    indexes = [t[0] for t in mean_npr_tuple]
    sem_npr = [sem_npr[i] for i in indexes]
    row_labels = [row_labels[i] for i in indexes]

    br1 = np.arange(len(mean_npr))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(br1, mean_npr, yerr=sem_npr)
    ax.set_xticks(np.arange(len(mean_npr)))
    ax.set_xticklabels(row_labels)
    _ = plt.setp(ax.get_xticklabels(), rotation=-20, ha="left",
                 rotation_mode="anchor")
    ax.set_ylabel("NPR")
    ax.yaxis.grid(True)
    if filename:
        fig.savefig(filename, bbox_inches="tight")