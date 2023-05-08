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
from gitk.eval.utils import load_genomic_embeddings
from sklearn.linear_model import LinearRegression
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["text.usetex"] = False
_log_path = None
GENOME_DIST_SCALAR = 1e10

def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename="log.txt"):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), "a") as f:
            f.write(obj)
            f.write("\n")

func_gdist = lambda u, v: float(u[1] < v[1]) * max(v[0] - u[1] + 1, 0) + float(
        u[1] >= v[1]
    ) * max(u[0] - v[1] + 1, 0)

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



def remap_name(name):
    return name.split('/')[-3]


def convert_position(pos):
    if pos // 1e6 > 0:
        return "{:.4f} MB".format(pos / 1e6)
    elif pos // 1e3 > 0:
        return "{:.4f} KB".format(pos / 1e3)
    else:
        return "{:.4f} B".format(pos)

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
    gdist_arr = [r[2]/GENOME_DIST_SCALAR for r in regions]
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