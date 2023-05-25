import glob
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from tqdm import tqdm

from .utils import load_genomic_embeddings, region2tuple


def random_annotate_points(
    labels, num_per_cluster=10, min_num_per_cluster=1, max_font_size=20, min_font_size=8
):
    cluster_labels = np.unique(labels)  # sorted
    positions = np.arange(len(labels))
    if cluster_labels[0] == -1:
        print(f"Number of clusters: {len(cluster_labels) - 1}")
    else:
        print(f"Number of clusters: {len(cluster_labels)}")
    clusters = [positions[labels == c] for c in cluster_labels]
    ratios = np.array([len(clusters[i]) / len(labels) for i in range(len(clusters))])
    annotate_arr = []
    for i, c in enumerate(cluster_labels):
        num = len(clusters[i])
        annotate_num = max(
            min(int(ratios[i] / ratios.max() * num_per_cluster), num),
            min(num, min_num_per_cluster),
        )
        fsize = max(int(ratios[i] / ratios.max() * max_font_size), min_font_size)
        indices = np.random.permutation(num)[0:annotate_num]
        pos = clusters[i][indices]
        annotate_arr.extend([(p, c, fsize) for p in pos])
    return annotate_arr


def assign_color_by_size(cmap_name, labels):
    cluster_labels = np.unique(labels)  # sorted
    cluster_sizes = [(c, (labels == c).sum()) for c in cluster_labels]
    cluster_sizes = sorted(cluster_sizes, key=lambda x: -x[1])
    color_mapping = {c: i for i, (c, s) in enumerate(cluster_sizes) if c != -1}
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, len(color_mapping)))
    # add outlier color
    color_mapping[-1] = len(color_mapping)
    colors = np.vstack([colors, [0.0, 0, 0, 1.0]])
    label_colors = [colors[color_mapping[l]] for l in labels]
    return label_colors


def get_cluster_regions(cluster_idx, labels, vocab, path):
    positions = np.arange(len(labels))
    indices = positions[labels == cluster_idx]
    regions = [region2tuple(vocab[i]) for i in indices]
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"cluster_{cluster_idx}.bed"), "w") as f:
        for chr_name, start, end in regions:
            f.write(f"{chr_name}\t{start}\t{end}\n")


def clustering(model_path, embed_type, n_clusters, save_folder, seed=0):
    np.random.seed(seed)
    embeds, vocab = load_genomic_embeddings(model_path, embed_type)

    clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(
        embeds
    )
    labels = clustering.labels_
    cluster_idxes = np.sort(np.unique(labels))
    for c in cluster_idxes:
        get_cluster_regions(c, labels, vocab, save_folder)
    with open(os.path.join(save_folder, "labels.pickle"), "wb") as f:
        pickle.dump(labels, f)


def cal_significance_val(pvals, threshold):
    num = (pvals < threshold).sum() + (pvals > 1 - threshold).sum()
    return num / len(pvals)


def get_scc_tss(
    model_path,
    embed_type,
    save_folder,
    Rscript_path,
    assembly,
    K_arr=[5, 20, 40],
    num_samples=1000,
    threshold=0.0001,
    num_workers=10,
    seed=0,
):
    for K in K_arr:
        target_folder = os.path.join(save_folder, f"Kmeans_{K}")
        clustering(model_path, embed_type, K, target_folder, seed=0)
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(
        [
            Rscript_path,
            f"{curr_folder}/permutation.R",
            "--assembly",
            assembly,
            "--num_workers",
            str(num_workers),
            "--path",
            save_folder,
            "--num_samples",
            str(num_samples),
        ]
    )
    scores = []
    for K in K_arr:
        target_folder = os.path.join(save_folder, f"Kmeans_{K}")
        tmp_files = glob.glob(os.path.join(target_folder, "cluster_*.bed"))
        # for tmp in tmp_files:
        #     os.remove(tmp)
        with open(os.path.join(target_folder, "pvals.txt"), "r") as f:
            pvals = f.readlines()
        pvals = np.array([float(p.strip()) for p in pvals])
        score = cal_significance_val(pvals, threshold)
        scores.append(score)
    print(model_path)
    print(
        "(K, CCSI): "
        + " ".join([f"({K_arr[i]},{scores[i]:.6f})" for i in range(len(K_arr))])
    )
    print("\n")
    return scores


def get_scc_tss_batch(
    batch,
    save_folder,
    Rscript_path,
    assembly,
    K_arr=[5, 20, 40, 60, 100],
    num_samples=1000,
    threshold=0.0001,
    num_workers=10,
    seed=0,
):
    scores_batch = []
    for i, (model_path, embed_type) in enumerate(batch):
        target_folder = os.path.join(save_folder, f"model_{i}")
        scores = get_scc_tss(
            model_path,
            embed_type,
            target_folder,
            Rscript_path,
            assembly,
            K_arr,
            num_samples,
            threshold,
            num_workers,
            seed,
        )
        scores_batch.append(scores)
    scores_batch = np.array(scores_batch)
    return scores_batch


def cct_tss_eval(
    batch,
    save_folder,
    Rscript_path,
    assembly,
    K_arr=[5, 20, 40, 60, 100],
    num_samples=1000,
    threshold=0.0001,
    num_runs=20,
    num_workers=10,
):
    for seed in range(num_runs):
        target_folder = os.path.join(save_folder, f"cct_tss_seed{seed}")
        scores_batch = get_scc_tss_batch(
            batch,
            target_folder,
            Rscript_path,
            assembly,
            K_arr,
            num_samples,
            threshold,
            num_workers,
            seed,
        )
        result_path = os.path.join(save_folder, f"cct_tss_seed{seed}.pickle")
        scores_batch = [
            (batch[i][0], scores_batch[i]) for i in range(len(scores_batch))
        ]
        with open(result_path, "wb") as f:
            pickle.dump(scores_batch, f)
    return scores_batch
