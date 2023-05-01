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
from sklearn.linear_model import LinearRegression
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["text.usetex"] = False
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



def sample_from_vocab(vocab, num_samples, seed=42):
    func_gdist = lambda u, v: float(u[1] < v[1]) * max(v[0] - u[1] + 1, 0) + float(
        u[1] >= v[1]
    ) * max(u[0] - v[1] + 1, 0)
    
    chr_probs = {}
    region_dict = {}
    num_vocab = len(vocab)
    # build stat from vocab
    for region in vocab:
        chr_str, position = region.split(':')
        chr_str = chr_str.strip()
        start, end = position.split('-')
        start = int(start.strip())
        end = int(end.strip())    
        chr_probs[chr_str] = chr_probs.get(chr_str, 0) + 1
        if chr_str in region_dict:
            region_dict[chr_str].append((start, end))
        else:
            region_dict[chr_str] = [(start, end)]
    total = sum([chr_probs[k] for k in chr_probs])
    chr_probs = [(k,chr_probs[k]/total) for k in chr_probs]
    
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
        sel_indexes = np.random.choice(len(regions),2, replace=False)
        r1, r2 = regions[sel_indexes[0]], regions[sel_indexes[1]]
        gdist = func_gdist(r1, r2)
        sampled_regions.append(('{}:{}-{}'.format(sel_chr, r1[0], r1[1]),'{}:{}-{}'.format(sel_chr, r2[0], r2[1]),gdist))
        count += 1
    return sampled_regions

def get_edc(path, embed_type, bin_path, num_samples=10000, seed=42, queue=None, worker_id=None, dist='cosine'):
    if dist == 'cosine':
        dist_func = lambda x,y: (1-((x/np.linalg.norm(x)) * (y/np.linalg.norm(y))).sum())/2
    elif dist == 'euclidean':
        dist_func = lambda x,y: np.linalg.norm(x-y)
    
    embed_rep, vocab = load_genomic_embeddings(path, embed_type)
    embed_bin, vocab_bin = load_genomic_embeddings(bin_path, 'base')
    regions = sample_from_vocab(vocab, num_samples, seed)
    region2idx = {r:i for i,r in enumerate(vocab)}
    region2idx_bin = {r:i for i,r in enumerate(vocab_bin)}
    gdist_arr = [r[2] for r in regions]
    edist_arr = np.array([dist_func(embed_rep[region2idx[t[0]]], embed_rep[region2idx[t[1]]]) for t in regions])
    edist_arr = (edist_arr - edist_arr.mean()) / edist_arr.std()
    edist_arr_bin = np.array([dist_func(embed_bin[region2idx_bin[t[0]]], embed_bin[region2idx_bin[t[1]]]) for t in regions])
    edist_arr_bin = (edist_arr_bin - edist_arr_bin.mean()) / edist_arr_bin.std()

    gd_arr = list(zip(edist_arr_bin, edist_arr))
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
    results = ['' for i in range(num)]
    while True:
        m = q.get()
        if m == "kill":
            break
        index = m[0]
        results[index] = (m[1],m[2])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    return results

def get_edc_batch(batch, bin_path, num_samples=10000, seed=42, save_path=None, num_workers=1, dist='cosine'):
    if num_workers <= 1:
        edc_arr = []
        for path, embed_type in batch:
            edc = get_edc(path, embed_type, bin_path, num_samples, seed, dist=dist)
            print(path, edc)
            edc_arr.append((path, edc))
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(edc_arr, f)
    else:
        manager = mp.Manager()
        queue = manager.Queue()
        with mp.Pool(processes=num_workers) as pool:
            writer = pool.apply_async(
                writer_multiprocessing, (save_path, len(batch), queue)
            )
            all_processes = []
            for i, (path, embed_type) in enumerate(batch):
                process = pool.apply_async(
                    get_edc,
                    (
                        path,
                        embed_type,
                        bin_path,
                        num_samples,
                        seed,
                        queue,
                        i,
                        dist
                    ),
                )
                all_processes.append(process)

            for process in all_processes:
                process.get()
            queue.put("kill")
            edc_arr = writer.get()
    return edc_arr



def edc_eval(batch, bin_path, num_runs=20, num_samples=1000, dist='cosine', save_folder=None, num_workers=10):
    results_seeds = []
    for seed in range(num_runs):
        print("----------------Run {}----------------".format(seed))
        save_path = (
            os.path.join(save_folder, "edc_eval_seed{}".format(seed+42))
            if save_folder
            else None
        )
        result_list = get_edc_batch(
            batch, bin_path, num_samples, seed, save_path, num_workers, dist
        )
        results_seeds.append(result_list)

    edc_res = [[] for i in range(len(batch))]
    for results in results_seeds:
        for i, res in enumerate(results):
            edc_res[i].append(res[1])
            assert res[0] == batch[i][0], "key == batch[i][0]"

    mean_edc = [np.array(r).mean() for r in edc_res]
    std_edc = [np.array(r).std() for r in edc_res]
    models = [t[0] for t in batch]
    for i in range(len(mean_edc)):
        print(
            "{}\n edc (std): {:.4f} ({:.4f}) \n".format(
                batch[i][0], mean_edc[i], std_edc[i]
            )
        )
    edc_arr = [(batch[i][0], edc_res[i]) for i in range(len(batch))]
    # edc_arr = list(zip(models,mean_edc,std_edc))
    return edc_arr


def get_edc_results(save_paths):
    with open(save_paths[0], "rb") as f:
        results = pickle.load(f)
    num = len(results)
    edc_res = [[] for i in range(num)]
    models = ['' for i in range(num)]
    for path in save_paths:
        with open(path, "rb") as f:
            results = pickle.load(f)
            for i, res in enumerate(results):
                edc_res[i].append(res[1])
                models[i] = res[0]
    # for bar-plot
    # mean_edc = [np.median(np.array(r)) for r in edc_res]
    # std_edc = [np.array(r).std() for r in edc_res]
    # edc_arr = list(zip(models,mean_edc,std_edc))

    # for box-plot
    edc_arr = [(models[i], edc_res[i]) for i in range(num)]
    return edc_arr

def remap_name(name):
    return name.split('/')[-3]

def plot_edc_arr(edc_arr, row_labels, filename=None):
    # yerr = None
    data = [g[1] for g in edc_arr]
    mean_edc = [(i,np.mean(np.array(d))) for i,d in enumerate(data)]
    mean_edc = sorted(mean_edc, key=lambda x:-x[1])
    indexes = [t[0] for t in mean_edc]
    mean_edc = [t[1] for t in mean_edc]
    std_edc = [np.array(data[i]).std() for i in indexes]
    sem_edc = [g/np.sqrt(len(data[0])) for g in std_edc]
    row_labels = [row_labels[i] for i in indexes]

    br1 = np.arange(len(mean_edc))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(br1, mean_edc, yerr=sem_edc)
    ax.set_xticks(np.arange(len(mean_edc)))
    ax.set_xticklabels(row_labels)
    _ = plt.setp(ax.get_xticklabels(), rotation=-20, ha="left",
                 rotation_mode="anchor")
    ax.set_ylabel("EDC")
    ax.yaxis.grid(True)

    if filename:
        fig.savefig(filename, bbox_inches="tight")

def convert_position(pos):
    if pos // 1e6 > 0:
        return "{:.4f} MB".format(pos / 1e6)
    elif pos // 1e3 > 0:
        return "{:.4f} KB".format(pos / 1e3)
    else:
        return "{:.4f} B".format(pos)



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

def get_gds_results(save_paths):
    with open(save_paths[0], "rb") as f:
        results = pickle.load(f)
    num = len(results)
    gds_res = [[] for i in range(num)]
    models = ['' for i in range(num)]
    for path in save_paths:
        with open(path, "rb") as f:
            results = pickle.load(f)
            for i, res in enumerate(results):
                gds_res[i].append(res[1])
                models[i] = res[0]
   
    gds_arr = [(models[i], gds_res[i]) for i in range(num)]
    return gds_arr

def get_gds(path, embed_type, num_samples=10000, seed=42, queue=None, worker_id=None, dist='cosine'):
    if dist == 'cosine':
        dist_func = lambda x,y: (1-((x/np.linalg.norm(x)) * (y/np.linalg.norm(y))).sum())/2
    elif dist == 'euclidean':
        dist_func = lambda x,y: np.linalg.norm(x-y)
    
    embed_rep, vocab = load_genomic_embeddings(path, embed_type)
    regions = sample_from_vocab(vocab, num_samples, seed)
    region2idx = {r:i for i,r in enumerate(vocab)}
    gdist_arr = [r[2]/1e8 for r in regions]
    edist_arr = np.array([dist_func(embed_rep[region2idx[t[0]]], embed_rep[region2idx[t[1]]]) for t in regions])
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
    results = ['' for i in range(num)]
    while True:
        m = q.get()
        if m == "kill":
            break
        index = m[0]
        results[index] = (m[1],m[2])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(results, f)
    return results

def get_gds_batch(batch, num_samples=10000, seed=42, save_path=None, num_workers=1, dist='cosine'):
    if num_workers <= 1:
        gds_arr = []
        for path, embed_type in batch:
            gds = get_gds(path, embed_type, num_samples, seed, dist=dist)
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
            writer = pool.apply_async(
                writer_multiprocessing, (save_path, len(batch), queue)
            )
            all_processes = []
            for i, (path, embed_type) in enumerate(batch):
                process = pool.apply_async(
                    get_gds,
                    (
                        path,
                        embed_type,
                        num_samples,
                        seed,
                        queue,
                        i,
                        dist
                    ),
                )
                all_processes.append(process)

            for process in all_processes:
                process.get()
            queue.put("kill")
            gds_arr = writer.get()
    return gds_arr

def gds_eval(batch, num_runs=20, num_samples=1000, dist='cosine', save_folder=None, num_workers=10):
    results_seeds = []
    for seed in range(num_runs):
        print("----------------Run {}----------------".format(seed))
        save_path = (
            os.path.join(save_folder, "gds_eval_seed{}".format(seed+42))
            if save_folder
            else None
        )
        result_list = get_gds_batch(
            batch, num_samples, seed, save_path, num_workers, dist
        )
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
        print(
            "{}\n GDS (std): {:.4f} ({:.4f}) \n".format(
                batch[i][0], mean_gds[i], std_gds[i]
            )
        )
    gds_arr = [(batch[i][0], gds_res[i]) for i in range(len(batch))]
    return gds_arr

def plot_gds_arr(gds_arr, row_labels, filename=None):
    # yerr = None
    data = [g[1] for g in gds_arr]
    mean_gds = [(i,np.mean(np.array(d))) for i,d in enumerate(data)]
    mean_gds = sorted(mean_gds, key=lambda x:-x[1])
    indexes = [t[0] for t in mean_gds]
    mean_gds = [t[1] for t in mean_gds]
    std_gds = [np.array(data[i]).std() for i in indexes]
    sem_gds = [g/np.sqrt(len(data[0])) for g in std_gds]
    row_labels = [row_labels[i] for i in indexes]

    br1 = np.arange(len(mean_gds))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(br1, mean_gds, yerr=sem_gds)
    ax.set_xticks(np.arange(len(mean_gds)))
    ax.set_xticklabels(row_labels)
    _ = plt.setp(ax.get_xticklabels(), rotation=-20, ha="left",
                 rotation_mode="anchor")
    ax.set_ylabel("GDS")
    ax.yaxis.grid(True)
    if filename:
        fig.savefig(filename, bbox_inches="tight")


# def gdt_box_plot(
#     ratio_data, r2_data, row_labels=None, legend_pos=(0.25, 0.6), filename=None
# ):
#     cmap = plt.get_cmap("Set1")
#     cmaplist = [cmap(i) for i in range(9)]
#     # sort based on the mean slope values
#     ratio_data = [
#         np.array(r) / np.sqrt(1 + np.array(r) * np.array(r)) for r in ratio_data
#     ]
#     mean_ratio = [(i, np.array(r).mean()) for i, r in enumerate(ratio_data)]
#     mean_ratio = sorted(mean_ratio, key=lambda x: -x[1])

#     indexes = [m[0] for m in mean_ratio]
#     mean_ratio = np.array([m[1] for m in mean_ratio])
#     pos_slope_indexes = np.array([i for i in indexes if mean_ratio[i] > 0])
#     neg_slope_indexes = np.array([i for i in indexes if mean_ratio[i] < 0])
#     std_ratio = np.array([np.array(ratio_data[i]).std() for i in indexes])
#     row_labels = [row_labels[i] for i in indexes]
#     r2_data = [r2_data[i] for i in indexes]
#     mean_r2s = np.array([np.array(e).mean() for e in r2_data])
#     std_r2s = np.array([np.array(e).std() for e in r2_data])
#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.plot(
#         range(len(mean_r2s)), mean_r2s, color=cmaplist[2], marker="o", linestyle="solid"
#     )
#     ax.fill_between(
#         range(len(mean_r2s)),
#         mean_r2s - std_r2s,
#         mean_r2s + std_r2s,
#         color=cmaplist[2],
#         alpha=0.1,
#     )
#     ax.set_xticks(list(range(len(mean_r2s))), labels=row_labels)
#     ax.set_ylabel(r"$R^2$")
#     _ = plt.setp(
#         ax.get_xticklabels(), rotation=-15, ha="left", va="top", rotation_mode="anchor"
#     )

#     ax1 = ax.twinx()
#     # ax1.fill_between(range(len(mean_r2s)), min(mean_ratio-std_ratio), 0.0, where=(mean_ratio<0), color=cmaplist[0],alpha=0.1)
#     ax1.plot(
#         range(len(mean_ratio)),
#         mean_ratio,
#         color=cmaplist[1],
#         marker="o",
#         linestyle="solid",
#     )
#     ax1.fill_between(
#         range(len(mean_ratio)),
#         mean_ratio - std_ratio,
#         mean_ratio + std_ratio,
#         color=cmaplist[1],
#         alpha=0.1,
#     )
#     ax1.set_ylabel(r"$sin(\alpha)$")

#     shaded_range = np.arange(len(mean_r2s))[mean_ratio < 0]
#     if len(shaded_range) > 0:
#         ax1.axvspan(
#             shaded_range.min(), shaded_range.max(), color=cmaplist[0], alpha=0.1
#         )
#         patches = [
#             Line2D(
#                 [0], [0], marker="o", linestyle="", color=cmaplist[1], markersize=10
#             ),
#             Patch(color=cmaplist[1], alpha=0.1),
#             Line2D(
#                 [0], [0], marker="o", linestyle="", color=cmaplist[2], markersize=10
#             ),
#             Patch(color=cmaplist[2], alpha=0.1),
#             Patch(color=cmaplist[0], alpha=0.1),
#         ]
#         legend = ax1.legend(
#             labels=[
#                 r"$sin(\alpha)$",
#                 r"std($sin(\alpha)$)",
#                 r"$R^2$",
#                 r"std($R^2$)",
#                 r"$sin(\alpha) < 0$",
#             ],
#             handles=patches,
#             bbox_to_anchor=legend_pos,
#             loc="center left",
#             borderaxespad=0,
#             fontsize=12,
#             frameon=True,
#         )

#     else:
#         patches = [
#             Line2D(
#                 [0], [0], marker="o", linestyle="", color=cmaplist[1], markersize=10
#             ),
#             Patch(color=cmaplist[1], alpha=0.1),
#             Line2D(
#                 [0], [0], marker="o", linestyle="", color=cmaplist[2], markersize=10
#             ),
#             Patch(color=cmaplist[2], alpha=0.1),
#         ]
#         legend = ax1.legend(
#             labels=[r"$sin(\alpha)$", r"std($sin(\alpha)$)", r"$R^2$", r"std($R^2$)"],
#             handles=patches,
#             bbox_to_anchor=legend_pos,
#             loc="center left",
#             borderaxespad=0,
#             fontsize=12,
#             frameon=True,
#         )

#     ax.grid("on")
#     if filename:
#         fig.savefig(filename, bbox_inches="tight")
#     return row_labels, mean_ratio, mean_r2s
