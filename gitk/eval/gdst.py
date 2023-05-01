import pickle
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import random
import glob
import time
import time
import multiprocessing as mp
import argparse
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from gitk.eval import load_base_embeddings

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            f.write(obj)
            f.write("\n")


def load_genomic_embeddings(model_path, embed_type="region2vec"):
    if embed_type == "region2vec":
        model = Word2Vec.load(model_path)
        regions_r2v = model.wv.index_to_key
        embed_rep = model.wv.vectors
        return embed_rep, regions_r2v
    elif embed_type == "base":
        embed_rep, regions_r2v = load_base_embeddings(model_path)
        return embed_rep, regions_r2v


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


# function calculating the chromosome distance between two regions
func_gdist = lambda u, v: float(u[1] < v[1]) * max(v[0] - u[1] + 1, 0) + float(
    u[1] >= v[1]
) * max(u[0] - v[1] + 1, 0)


def embed_distance(x1, x2, metric):
    if metric == "cosine":
        n1 = np.linalg.norm(x1)
        n2 = np.linalg.norm(x2)
        dist = 1 - np.dot(x1 / n1, x2 / n2)
    elif metric == "euclidean":
        dist = np.linalg.norm(x1 - x2)
    else:
        raise ("Invalid metric function")
    return dist


def sample_pair(chromo_regions, chromo_ratios):
    chromo_arr = [t[0] for t in chromo_ratios]
    probs = [t[1] for t in chromo_ratios]
    chromo = np.random.choice(chromo_arr, p=probs)
    region_arr = chromo_regions[chromo]

    idx1 = np.random.randint(len(region_arr))
    idx2 = np.random.randint(len(region_arr))
    while idx1 == idx2:
        idx2 = np.random.randint(len(region_arr))
    gdist = func_gdist(region_arr[idx1], region_arr[idx2])
    return chromo, idx1, idx2, gdist


def bin_search(boundaries, val):
    left = 0
    right = len(boundaries) - 1
    if val < boundaries[left] or val > boundaries[right]:
        return -1
    while left < right:
        mid = int((left + right) / 2)
        if boundaries[mid] == val:
            return mid - 1
        elif boundaries[mid] > val:
            right = mid
        else:
            left = mid + 1
    return left - 1


def fill_bins_via_sampling(
    embed_rep,
    embed_bin,
    regions_vocab,
    boundaries,
    num_per_bin,
    dist_metric,
    sum_statistic,
    seed,
):
    np.random.seed(seed)
    num, dim = embed_rep.shape

    embed_rep_ref = (np.random.rand(num, dim) - 0.5) / dim
    region2index = {r: i for i, r in enumerate(regions_vocab)}
    # Group regions by chromosomes
    chromo_regions = {}
    embed_dict = {}
    for i, v in enumerate(regions_vocab):
        chromo, region = v.split(":")  # e.g. chr1:100-1000
        chromo = chromo.strip()  # remove possible spaces
        region = region.strip()  # remove possible spaces
        start, end = region.split("-")
        start = int(start.strip())
        end = int(end.strip())
        if chromo_regions.get(chromo, None) is None:
            chromo_regions[chromo] = [(start, end)]
            embed_dict[chromo] = [i]
        else:
            chromo_regions[chromo].append((start, end))
            embed_dict[chromo].append(i)

    chromo_ratios = []
    for i, chromo in enumerate(chromo_regions):
        chromo_ratios.append((chromo, len(chromo_regions[chromo]) / len(regions_vocab)))

    num_bins = len(boundaries) - 1
    groups = [[] for i in range(num_bins)]
    counts = np.array([0 for i in range(num_bins)])
    overlaps = np.array([0 for i in range(num_bins)])
    total_samples = num_per_bin * num_bins
    num_try = 0
    MAX_TRY_NUMBER = 1e7
    while counts.sum() < total_samples:
        while True:
            num_try += 1
            chromo, idx1, idx2, gdist = sample_pair(chromo_regions, chromo_ratios)
            bin_idx = bin_search(boundaries, gdist)
            if bin_idx == -1:
                continue
            if counts[bin_idx] < num_per_bin:
                break
            if num_try >= MAX_TRY_NUMBER:
                break
        if num_try >= MAX_TRY_NUMBER:
            break
        emb_arr = embed_dict[chromo]
        eidx1, eidx2 = emb_arr[idx1], emb_arr[idx2]
        edist = embed_distance(embed_rep[eidx1], embed_rep[eidx2], dist_metric)
        overlap_ratio = (
            embed_bin[eidx1] * embed_bin[eidx2]
        ).sum()  # /embed_bin.shape[1]
        edist_ref = embed_distance(
            embed_rep_ref[eidx1], embed_rep_ref[eidx2], dist_metric
        )
        groups[bin_idx].append((gdist, edist, edist_ref))
        counts[bin_idx] += 1
        overlaps[bin_idx] += overlap_ratio
    records = []
    for i in range(num_bins):
        if counts[i] == 0:
            avg_gd = -1
            avg_ed = -1
            avg_ed_ref = -1
        else:
            if sum_statistic == "mean":
                avg_gd = np.array([t[0] for t in groups[i]]).mean()
                avg_ed = np.array([t[1] for t in groups[i]]).mean()
                avg_ed_ref = np.array([t[2] for t in groups[i]]).mean()
            elif sum_statistic == "median":
                avg_gd = np.median(np.array([t[0] for t in groups[i]]))
                avg_ed = np.median(np.array([t[1] for t in groups[i]]))
                avg_ed_ref = np.median(np.array([t[2] for t in groups[i]]))
        records.append((avg_gd, avg_ed, avg_ed_ref, counts[i], overlaps[i] / counts[i]))
    return records


def convert_position(pos):
    if pos // 1e6 > 0:
        return "{:.4f} MB".format(pos / 1e6)
    elif pos // 1e3 > 0:
        return "{:.4f} KB".format(pos / 1e3)
    else:
        return "{:.4f} B".format(pos)


def get_slope(avgGD, avgED, log_xscale=False):
    x = avgGD
    x1 = x[x > 0] / 1e8
    y = avgED
    y1 = y[x > 0]
    if log_xscale:
        x1 = np.log10(x1)
    A = np.vstack([x1, np.ones(len(x1))]).T
    lin_res = np.linalg.lstsq(A, y1, rcond=None)
    m, c = lin_res[0]  # slope, bias
    r = lin_res[1][0]  # approximation error
    r2 = r2_score(y1, m * x1 + c)
    return m, c, r2, x1, y1


def genome_distance_test(
    path,
    embed_type,
    boundaries,
    num_samples=100,
    metric="euclidean",
    sum_statistic="mean",
    seed=0,
    queue=None,
    worker_id=None,
):
    embed_rep, regions_vocab = load_genomic_embeddings(path, embed_type)
    bin_embed_path = os.path.join("/".join(path.split("/")[0:-3]), "bin_embed.pickle")
    embed_bin, regions_bin = load_base_embeddings(bin_embed_path)
    r2i = {r: i for i, r in enumerate(regions_bin)}
    embed_bin = np.array([embed_bin[r2i[r]] for r in regions_vocab])
    res = fill_bins_via_sampling(
        embed_rep,
        embed_bin,
        regions_vocab,
        boundaries,
        num_samples,
        metric,
        sum_statistic,
        seed,
    )
    msg1 = " ".join(["{:.4f}".format(r[0]) for r in res])
    msg2 = " ".join(["{:.4f}".format(r[1]) for r in res])
    msg3 = " ".join(["{:.4f}".format(r[2]) for r in res])
    msg4 = " ".join(["{:d}".format(r[3]) for r in res])
    msg5 = " ".join(["{:.4f}".format(r[4]) for r in res])

    res_dict = {
        "AvgGD": np.array([r[0] for r in res]),
        "AvgED": np.array([r[1] for r in res]),
        "AvgED_rand": np.array([r[2] for r in res]),
        "num_samples": np.array([r[3] for r in res]),
        "overlaps": np.array([r[4] for r in res]),
    }
    slope, bias, r2, x, y = get_slope(res_dict["AvgGD"], res_dict["AvgED"])
    res_dict["Slope"] = slope
    res_dict["R2"] = r2
    res_dict["Path"] = path
    msg = "[seed {}]: {}\n".format(seed, path)
    msg += (
        "AvgGD: "
        + msg1
        + "\n"
        + "AvgED: "
        + msg2
        + "\n"
        + "Slope: {:.4f} R2: {:.4f}\n".format(slope, r2)
        + "AvgED(random): "
        + msg3
        + "\n"
        + "Num Samples:"
        + msg4
        + "\n"
        + "Overlaped files:"
        + msg5
        + "\n"
    )
    print(msg)
    if queue:
        queue.put((worker_id, res_dict))
        return worker_id, res_dict, msg
    else:
        return res_dict


def genome_distance_test_batch(
    batch,
    boundaries,
    num_samples=100,
    metric="euclidean",
    sum_statistic="mean",
    seed=0,
    num_workers=5,
    save_path=None,
):
    timer = Timer()
    if num_workers <= 1:
        res_list = []
        for path, embed_type in batch:
            _, res, msg = genome_distance_test(
                path, embed_type, boundaries, num_samples, metric, sum_statistic, seed
            )
            res_list.append(res)
    else:  ## Multi-processing
        manager = mp.Manager()
        queue = manager.Queue()
        with mp.Pool(processes=num_workers) as pool:
            writer = pool.apply_async(
                writer_multiprocessing, (save_path, len(batch), queue)
            )
            all_processes = []
            for i, (path, embed_type) in enumerate(batch):
                process = pool.apply_async(
                    genome_distance_test,
                    (
                        path,
                        embed_type,
                        boundaries,
                        num_samples,
                        metric,
                        sum_statistic,
                        seed,
                        queue,
                        i,
                    ),
                )
                all_processes.append(process)

            for process in all_processes:
                process.get()
            queue.put("kill")
            res_list = writer.get()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(res_list, f)
    time_str = timer.measure()
    print("Finished. Elasped time: " + time_str)
    return res_list


def gdt_plot_fitted(avgGD, avgED, filename=None):
    # plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ratio, bias, r2, x, y = get_slope(avgGD, avgED)
    ax.plot(x, y, "-^")
    ax.plot(x, np.array(x) * ratio + bias, "r--")
    t = ax.text(
        0.48,
        0.85,
        "AvgGD={:.4f}*AvgED+{:.4f}".format(ratio, bias),
        ha="center",
        va="center",
        size=15,
        transform=ax.transAxes,
    )
    ax.set_xlabel(r"AvgGD ($10^8$)")
    ax.set_ylabel(r"AvgED")
    if filename:
        fig.savefig(filename, bbox_inches="tight")


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=12)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=12)

    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=12)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    fontsize=10,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(
        horizontalalignment="center", verticalalignment="center", fontsize=fontsize
    )
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if im.norm(data[i, j]) > 0.9 or im.norm(data[i, j]) < 0.1:
                # if im.norm(data[i, j]) > 0.5:
                index = 1
            else:
                index = 0
            kw.update(color=textcolors[index])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_gdt_results(save_paths):
    with open(save_paths[0], "rb") as f:
        results = pickle.load(f)
    num = len(results)
    r2_res = [[] for i in range(num)]
    ratio_res = [[] for i in range(num)]
    avg_gd_res = [[] for i in range(num)]
    avg_ed_res = [[] for i in range(num)]
    row_labels = ["" for i in range(num)]
    for path in save_paths:
        with open(path, "rb") as f:
            results = pickle.load(f)
            for i, res in enumerate(results):
                key = res["Path"]
                slope = res["Slope"]
                r2 = res["R2"]
                r2_res[i].append(r2)
                ratio_res[i].append(slope)
                avg_gd_res[i].append(res["AvgGD"])
                avg_ed_res[i].append(res["AvgED"])
                row_labels[i] = key.split("/")[-3]
    avg_gd_res = np.array(avg_gd_res).mean(axis=1)
    avg_ed_res = np.array(avg_ed_res).mean(axis=1)
    mean_ratios = np.array(ratio_res).mean(axis=1)
    mean_tuple = sorted(
        [(i, m) for i, m in enumerate(mean_ratios)], key=lambda x: -x[1]
    )
    indexes = [t[0] for t in mean_tuple]
    mean_ratios = [t[1] for t in mean_tuple]
    avg_ed_res = np.array([avg_ed_res[i] for i in indexes])
    row_labels = [row_labels[i] for i in indexes]
    fig, ax = plt.subplots(figsize=(10, 20))
    im, cbar = heatmap(
        avg_ed_res,
        row_labels,
        ["Group1", "Group2", "Group3", "Group4"],
        ax=ax,
        cmap="RdYlBu",
        cbarlabel="AvgED",
    )
    texts = annotate_heatmap(im, valfmt="{x:.2f}", fontsize=12)
    fig.savefig("avged_heatmap.png")
    return ratio_res, r2_res


def gdt_eval(batch, boundaries, num_runs=20, num_samples=1000, save_folder=None):
    results_seeds = []
    for seed in range(num_runs):
        print("----------------Run {}----------------".format(seed))
        save_path = (
            os.path.join(save_folder, "gdt_eval_seed{}".format(seed))
            if save_folder
            else None
        )
        result_list = genome_distance_test_batch(
            batch, boundaries, num_samples=num_samples, seed=seed, save_path=save_path
        )
        results_seeds.append(result_list)

    # get average slopes and R2 values for the two models
    r2_res = [[] for i in range(len(batch))]
    ratio_res = [[] for i in range(len(batch))]
    for results in results_seeds:
        for i, res in enumerate(results):
            key = res["Path"]
            slope = res["Slope"]
            r2 = res["R2"]
            r2_res[i].append(r2)
            ratio_res[i].append(slope)
            assert key == batch[i][0], "key == batch[i][0]"

    mean_ratio = [np.array(r).mean() for r in ratio_res]
    std_ratio = [np.array(r).std() for r in ratio_res]

    mean_r2 = [np.array(e).mean() for e in r2_res]
    std_r2 = [np.array(e).std() for e in r2_res]

    for i in range(len(mean_ratio)):
        print(
            "{}\n Slope (std): {:.4f} ({:.4f}) | R2 (std): {:.4f} ({:.4f}) \n".format(
                batch[i][0], mean_ratio[i], std_ratio[i], mean_r2[i], std_r2[i]
            )
        )
    return ratio_res, r2_res


def writer_multiprocessing(save_path, num, q):
    results = [[] for i in range(num)]
    while True:
        m = q.get()
        if m == "kill":
            break
        index = m[0]
        results[index] = m[1]
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    return results


def gdt_box_plot(
    ratio_data, r2_data, row_labels=None, legend_pos=(0.25, 0.6), filename=None
):
    cmap = plt.get_cmap("Set1")
    cmaplist = [cmap(i) for i in range(9)]
    # sort based on the mean slope values
    ratio_data = [
        np.array(r) / np.sqrt(1 + np.array(r) * np.array(r)) for r in ratio_data
    ]
    mean_ratio = [(i, np.array(r).mean()) for i, r in enumerate(ratio_data)]
    mean_ratio = sorted(mean_ratio, key=lambda x: -x[1])

    indexes = [m[0] for m in mean_ratio]
    mean_ratio = np.array([m[1] for m in mean_ratio])
    pos_slope_indexes = np.array([i for i in indexes if mean_ratio[i] > 0])
    neg_slope_indexes = np.array([i for i in indexes if mean_ratio[i] < 0])
    std_ratio = np.array([np.array(ratio_data[i]).std() for i in indexes])
    row_labels = [row_labels[i] for i in indexes]
    r2_data = [r2_data[i] for i in indexes]
    mean_r2s = np.array([np.array(e).mean() for e in r2_data])
    std_r2s = np.array([np.array(e).std() for e in r2_data])
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        range(len(mean_r2s)), mean_r2s, color=cmaplist[2], marker="o", linestyle="solid"
    )
    ax.fill_between(
        range(len(mean_r2s)),
        mean_r2s - std_r2s,
        mean_r2s + std_r2s,
        color=cmaplist[2],
        alpha=0.1,
    )
    ax.set_xticks(list(range(len(mean_r2s))), labels=row_labels)
    ax.set_ylabel(r"$R^2$")
    _ = plt.setp(
        ax.get_xticklabels(), rotation=-15, ha="left", va="top", rotation_mode="anchor"
    )

    ax1 = ax.twinx()
    # ax1.fill_between(range(len(mean_r2s)), min(mean_ratio-std_ratio), 0.0, where=(mean_ratio<0), color=cmaplist[0],alpha=0.1)
    ax1.plot(
        range(len(mean_ratio)),
        mean_ratio,
        color=cmaplist[1],
        marker="o",
        linestyle="solid",
    )
    ax1.fill_between(
        range(len(mean_ratio)),
        mean_ratio - std_ratio,
        mean_ratio + std_ratio,
        color=cmaplist[1],
        alpha=0.1,
    )
    ax1.set_ylabel(r"$sin(\alpha)$")

    shaded_range = np.arange(len(mean_r2s))[mean_ratio < 0]
    if len(shaded_range) > 0:
        ax1.axvspan(
            shaded_range.min(), shaded_range.max(), color=cmaplist[0], alpha=0.1
        )
        patches = [
            Line2D(
                [0], [0], marker="o", linestyle="", color=cmaplist[1], markersize=10
            ),
            Patch(color=cmaplist[1], alpha=0.1),
            Line2D(
                [0], [0], marker="o", linestyle="", color=cmaplist[2], markersize=10
            ),
            Patch(color=cmaplist[2], alpha=0.1),
            Patch(color=cmaplist[0], alpha=0.1),
        ]
        legend = ax1.legend(
            labels=[
                r"$sin(\alpha)$",
                r"std($sin(\alpha)$)",
                r"$R^2$",
                r"std($R^2$)",
                r"$sin(\alpha) < 0$",
            ],
            handles=patches,
            bbox_to_anchor=legend_pos,
            loc="center left",
            borderaxespad=0,
            fontsize=12,
            frameon=True,
        )

    else:
        patches = [
            Line2D(
                [0], [0], marker="o", linestyle="", color=cmaplist[1], markersize=10
            ),
            Patch(color=cmaplist[1], alpha=0.1),
            Line2D(
                [0], [0], marker="o", linestyle="", color=cmaplist[2], markersize=10
            ),
            Patch(color=cmaplist[2], alpha=0.1),
        ]
        legend = ax1.legend(
            labels=[r"$sin(\alpha)$", r"std($sin(\alpha)$)", r"$R^2$", r"std($R^2$)"],
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
    return row_labels, mean_ratio, mean_r2s
