import pickle
import numpy as np
import glob
import os
import time
from sklearn.cluster import KMeans
from tqdm import tqdm
import shutil
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing as mp
from gitk.eval.utils import load_genomic_embeddings
import subprocess


def random_annotate_points(
    labels, num_per_cluster=10, min_num_per_cluster=1, max_font_size=20, min_font_size=8
):
    cluster_labels = np.unique(labels)  # sorted
    positions = np.arange(len(labels))
    if cluster_labels[0] == -1:
        print("Number of clusters: {}".format(len(cluster_labels) - 1))
    else:
        print("Number of clusters: {}".format(len(cluster_labels)))
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
    def region2tuple(x):
        eles = x.split(":")
        chr_name = eles[0].strip()
        start, end = eles[1].split("-")
        start, end = int(start.strip()), int(end.strip())
        return chr_name, start, end

    positions = np.arange(len(labels))
    indices = positions[labels == cluster_idx]
    regions = [region2tuple(vocab[i]) for i in indices]
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "cluster_{}.bed".format(cluster_idx)), "w") as f:
        for chr_name, start, end in regions:
            f.write("{}\t{}\t{}\n".format(chr_name, start, end))


def clustering(model_path, embed_type, K, save_folder, seed=0):
    np.random.seed(seed)
    embeds, vocab = load_genomic_embeddings(model_path, embed_type)

    clustering = KMeans(n_clusters=K, random_state=seed, n_init="auto").fit(embeds)
    labels = clustering.labels_
    cluster_idxes = np.sort(np.unique(labels))
    for c in cluster_idxes:
        get_cluster_regions(c, labels, vocab, save_folder)
    with open(os.path.join(save_folder, "labels.pickle"), "wb") as f:
        pickle.dump(labels, f)


def clustering_batch(batch, K, save_folder, seed=0, num_workers=10):
    worker_func = clustering
    with mp.Pool(processes=num_workers) as pool:
        all_processes = []
        for i, (path, embed_type) in enumerate(batch):
            folder = os.path.join(save_folder, "model_{}".format(i))
            os.makedirs(folder, exist_ok=True)
            process = pool.apply_async(worker_func, (path, embed_type, K, folder))
            all_processes.append(process)
        for process in all_processes:
            process.get()


def cal_significance_val(pvals, threshold):
    num = (pvals < threshold).sum() + (pvals > 1 - threshold).sum()
    return num / len(pvals)


def get_scctss(
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
        target_folder = os.path.join(save_folder, "Kmeans_{}".format(K))
        clustering(model_path, embed_type, K, target_folder, seed=0)
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    subprocess.call(
        [
            Rscript_path,
            "{}/permutation.R".format(curr_folder),
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
        target_folder = os.path.join(save_folder, "Kmeans_{}".format(K))
        tmp_files = glob.glob(os.path.join(target_folder, "cluster_*.bed"))
        for tmp in tmp_files:
            os.remove(tmp)
        with open(os.path.join(target_folder, "pvals.txt"), "r") as f:
            pvals = f.readlines()
        pvals = np.array([float(p.strip()) for p in pvals])
        score = cal_significance_val(pvals, threshold)
        scores.append(score)
    print(model_path)
    print(
        "(K, CCSI): "
        + " ".join(
            ["({},{:.6f})".format(K_arr[i], scores[i]) for i in range(len(K_arr))]
        )
    )
    print("\n")
    return scores


def get_scctss_batch(
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
        target_folder = os.path.join(save_folder, "model_{}".format(i))
        scores = get_scctss(
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
    avg_ranks = rankdata(-scores_batch, method="average", axis=0).mean(axis=1)
    return scores_batch, avg_ranks


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
    avg_ranks_arr = []
    for seed in range(num_runs):
        target_folder = os.path.join(save_folder, "cct_tss_seed{}".format(seed))
        scores_batch, avg_ranks = get_scctss_batch(
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
        avg_ranks_arr.append(avg_ranks)
        result_path = os.path.join(save_folder, "cct_tss_seed{}.pickle".format(seed))
        scores_batch = [
            (batch[i][0], scores_batch[i]) for i in range(len(scores_batch))
        ]
        with open(result_path, "wb") as f:
            pickle.dump(scores_batch, f)
    avg_ranks_arr = np.vstack(avg_ranks_arr)
    avg_ranks_arr = [(batch[i][0], avg_ranks_arr[:, i]) for i in range(num_runs)]
    return avg_ranks_arr


def cct_tss_plot(avg_ranks_arr, row_labels=None, legend_pos=(0.25, 0.6), filename=None):
    mean_rank = [t[1].mean() for t in avg_ranks_arr]
    std_rank = [t[1].std() for t in avg_ranks_arr]
    mean_rank_tuple = [(i, r) for i, r in enumerate(mean_rank)]
    mean_rank_tuple = sorted(mean_rank_tuple, key=lambda x: x[1])
    indexes = [t[0] for t in mean_rank_tuple]

    if row_labels is None:
        row_labels = [t[0] for t in avg_ranks_arr]

    mean_rank = [mean_rank[i] for i in indexes]
    std_rank = [std_rank[i] for i in indexes]
    row_labels = [row_labels[i] for i in indexes]

    cmap = plt.get_cmap("Set1")
    cmaplist = [cmap(i) for i in range(9)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xticks(list(range(1, len(mean_rank) + 1)))

    ax.errorbar(
        range(1, len(mean_rank) + 1),
        mean_rank,
        yerr=std_rank,
        fmt="o",
        ms=10,
        mfc=cmaplist[1],
        mec=cmaplist[8],
        ecolor=cmaplist[2],
        elinewidth=3,
        capsize=5,
    )
    ax.set_xticklabels(row_labels)
    ax.set_ylabel("CCSI Rank")
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
        labels=["CCSI average rank", "CCSI rank standard deviation"],
        handles=patches,
        bbox_to_anchor=legend_pos,
        loc="center left",
        borderaxespad=0,
        fontsize=12,
        frameon=True,
    )
    ax.grid("on")
    ax.set_ylim(ax.get_ylim()[::-1])
    if filename:
        fig.savefig(filename, bbox_inches="tight")
