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
from matplotlib.lines import Line2D
import matplotlib
from matplotlib.patches import Patch

def load_genomic_embeddings(model_path, embed_type='region2vec'):
    if embed_type == 'region2vec':
        model = Word2Vec.load(model_path)
        regions_r2v = model.wv.index2word
        embed_rep = model.wv.vectors
        return embed_rep, regions_r2v
    elif embed_type == 'base':
        embed_rep, regions_r2v = load_base_embeddings(model_path)
        return embed_rep, regions_r2v

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

# function calculating the chromosome distance between two regions
func_gdist = lambda u,v:float(u[1]<v[1])*max(v[0]-u[1]+1,0)+float(u[1]>=v[1])*max(u[0]-v[1]+1,0)


def embed_distance(x1, x2, metric):
    if metric == 'cosine':
        n1 = np.linalg.norm(x1)
        n2 = np.linalg.norm(x2)
        dist = 1 - np.dot(x1/n1, x2/n2)
    elif metric == 'euclidean':
        dist = np.linalg.norm(x1-x2)
    else:
        raise('Invalid metric function')
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
        mid = int((left + right)/2)
        if boundaries[mid] == val:
            return mid - 1
        elif boundaries[mid] > val:
            right = mid
        else:
            left = mid + 1
    return left - 1

def fill_bins_via_sampling(embed_rep, regions_vocab, boundaries, num_per_bin, dist_metric, sum_statistic, seed):
    np.random.seed(seed)
    num, dim = embed_rep.shape
    
    embed_rep_ref = (np.random.rand(num,dim)-0.5)/dim
    region2index = {r:i for i,r in enumerate(regions_vocab)}
    # Group regions by chromosomes
    chromo_regions = {}
    embed_dict = {}
    for i,v in enumerate(regions_vocab):
        chromo, region = v.split(':') # e.g. chr1:100-1000
        chromo = chromo.strip() # remove possible spaces
        region = region.strip() # remove possible spaces
        start, end = region.split('-')
        start = int(start.strip())
        end = int(end.strip())
        if chromo_regions.get(chromo, None) is None:
            chromo_regions[chromo] = [(start,end)]
            embed_dict[chromo] = [i]
        else:
            chromo_regions[chromo].append((start,end))
            embed_dict[chromo].append(i)
            
    chromo_ratios = []
    for i,chromo in enumerate(chromo_regions):
        chromo_ratios.append((chromo,len(chromo_regions[chromo])/len(regions_vocab)))

    
    num_bins = len(boundaries) - 1
    groups = [[] for i in range(num_bins)]
    counts = np.array([0 for i in range(num_bins)])
    total_samples = num_per_bin*num_bins
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
        edist_ref = embed_distance(embed_rep_ref[eidx1], embed_rep_ref[eidx2], dist_metric)
        groups[bin_idx].append((gdist, edist, edist_ref))
        counts[bin_idx] += 1
    records = []
    for i in range(num_bins):
        if counts[i] == 0:
            avg_gd = -1
            avg_ed = -1
            avg_ed_ref = -1
        else:
            if sum_statistic == 'mean':
                avg_gd = np.array([t[0] for t in groups[i]]).mean()
                avg_ed = np.array([t[1] for t in groups[i]]).mean()
                avg_ed_ref = np.array([t[2] for t in groups[i]]).mean()
            elif sum_statistic == 'median':
                avg_gd = np.median(np.array([t[0] for t in groups[i]]))
                avg_ed = np.median(np.array([t[1] for t in groups[i]]))
                avg_ed_ref = np.median(np.array([t[2] for t in groups[i]]))
        records.append((avg_gd, avg_ed, avg_ed_ref,counts[i]))
    return records


def convert_position(pos):
    if pos // 1e6 > 0:
        return '{:.4f} MB'.format(pos/1e6)
    elif pos // 1e3 > 0:
        return '{:.4f} KB'.format(pos/1e3)
    else:
        return '{:.4f} B'.format(pos)

def get_slope(avgGD, avgED, log_xscale=False):
    x = avgGD
    x1 = x[x>0]/1e8
    y = avgED
    y1 = y[x>0]
    if log_xscale:
        x1 = np.log10(x1)
    A = np.vstack([x1, np.ones(len(x1))]).T
    lin_res = np.linalg.lstsq(A, y1, rcond=None)
    m, c = lin_res[0] # slope, bias
    r = lin_res[1][0] # approximation error
    return m, c, r, x1, y1


def genome_distance_test(path, embed_type, boundaries, num_samples=100, metric='euclidean', sum_statistic='mean', seed=0, queue=None, worker_id=None):
    embed_rep, regions_vocab = load_genomic_embeddings(path, embed_type)
    res = fill_bins_via_sampling(embed_rep, regions_vocab,boundaries,num_samples,metric,sum_statistic,seed)
    msg1 =  ' '.join(['{:.4f}'.format(r[0]) for r in res])
    msg2 =  ' '.join(['{:.4f}'.format(r[1]) for r in res])
    msg3 =  ' '.join(['{:.4f}'.format(r[2]) for r in res])
    msg4 =  ' '.join(['{:d}'.format(r[3]) for r in res])
    
    
    res_dict = {'AvgGD':np.array([r[0] for r in res]), 'AvgED':np.array([r[1] for r in res]), 'AvgED_rand':np.array([r[2] for r in res]), 'num_samples':np.array([r[3] for r in res])}
    slope, bias, err, x, y = get_slope(res_dict['AvgGD'], res_dict['AvgED'])
    res_dict['Slope'] = slope
    res_dict['Error'] = err
    res_dict['Path'] = path
    msg = '[seed {}]'.format(seed)
    msg += 'AvgGD: ' + msg1 + '\n' + 'AvgED: ' + msg2 + '\n' + 'Slope: {:.4f} Approx. Err: {:.4f}\n'.format(slope,err) + 'AvgED(random): ' + msg3 + '\n' + 'Num Samples:' + msg4 +'\n'
    print(msg)
    if queue:
        queue.put((worker_id, res_dict))
        return worker_id, res_dict, msg
    else:
        return res_dict



def genome_distance_test_batch(batch, boundaries, num_samples=100, metric='euclidean', sum_statistic='mean', seed=0, num_workers=5, save_path=None):
    timer = Timer()
    if num_workers <= 1:
        res_list = []
        for path, embed_type in batch:
            _, res, msg = genome_distance_test(path, embed_type, boundaries, num_samples, metric, sum_statistic, seed)
            res_list.append(res)
    else: ## Multi-processing
        manager = mp.Manager()
        queue = manager.Queue()    
        with mp.Pool(processes=num_workers) as pool:
            writer = pool.apply_async(writer_multiprocessing, (save_path, len(batch), queue))
            all_processes = []
            for i, (path, embed_type) in enumerate(batch):
                process = pool.apply_async(genome_distance_test, (path, embed_type, boundaries, num_samples, metric, sum_statistic, seed, queue, i))
                all_processes.append(process)
        
            for process in all_processes:
                process.get()
            queue.put('kill')
            res_list = writer.get()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(res_list, f)
    time_str = timer.measure()
    print('Finished. Elasped time: ' + time_str)
    return res_list

def gdt_plot_fitted(avgGD, avgED, filename=None):
    # plt.rcParams['text.usetex'] = True
    fig, ax  = plt.subplots(figsize=(5,2.5))
    ratio, bias, err, x, y = get_slope(avgGD, avgED)
    ax.plot(x,y,'-^')
    ax.plot(x, np.array(x)*ratio+bias,'r--')
    t = ax.text(0.48, 0.85, "AvgGD={:.4f}*AvgED+{:.4f}".format(ratio,bias), ha="center", va="center", size=15,transform=ax.transAxes)
    ax.set_xlabel(r'AvgGD ($10^8$)')
    ax.set_ylabel(r'AvgED')
    if filename:
        fig.savefig(filename,bbox_inches='tight')


def get_gdt_results(save_paths):
    err_res = {}
    ratio_res = {}
    for path in save_paths:
        with open(path, 'rb') as f:
            results = pickle.load(f)
            for res in results:
                key = res['Path']
                slope = res['Slope']
                err = res['Error']
                if key in err_res:
                    err_res[key].append(err)
                    ratio_res[key].append(slope)
                else:
                    err_res[key] = [err]
                    ratio_res[key] = [slope]
    return ratio_res, err_res

def gdt_eval(batch, boundaries, num_runs=20, num_samples=1000, save_folder=None):
    results_seeds = []
    for seed in range(num_runs):
        print('----------------Run {}----------------'.format(seed))
        save_path = os.path.join(save_folder,'gdt_eval_seed{}'.format(seed)) if save_folder else None
        result_list = genome_distance_test_batch(batch, boundaries, num_samples=num_samples, seed=seed, save_path=save_path)
        results_seeds.append(result_list)

    # get average slopes and approximation errors for the two models
    err_res = [[] for i in range(len(batch))]
    ratio_res = [[] for i in range(len(batch))]
    paths = ['' for i in range(len(batch))]
    for results in results_seeds:
        for i, res in enumerate(results):
            key = res['Path']
            slope = res['Slope']
            err = res['Error']
            err_res[i].append(err)
            ratio_res[i].append(slope)
            paths[i] = key
    mean_ratio = [np.array(r).mean() for r in ratio_res]
    std_ratio = [np.array(r).std() for r in ratio_res]

    mean_err = [np.array(e).mean() for e in err_res]
    std_err = [np.array(e).std() for e in err_res]

    ratio_res = [(paths[i],ratio_res[i]) for i in range(len(ratio_res))]
    err_res = [(paths[i],err_res[i]) for i in range(len(err_res))]
    for i in range(len(mean_ratio)):
        print('{}\n Slope (std): {:.4f} ({:.4f}) | ApproxErr (std): {:.4f} ({:.4f}) \n'.format(paths[i], mean_ratio[i], std_ratio[i], mean_err[i], std_err[i]))
    return ratio_res, err_res

def writer_multiprocessing(save_path, num, q):
    results = [[] for i in range(num)]
    while True:
        m = q.get()
        if m == 'kill':
            break
        index = m[0]
        results[index] = m[1]
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
    return results

def gdt_box_plot(ratio_data, err_data, row_lables=None, legend_pos=(0.25, 0.6), filename=None):
    rdata = [v for k, v in ratio_data]
    edata = [v for k, v in err_data]
    cmap = plt.get_cmap('Set1')
    cmaplist = [cmap(i) for i in range(9)]
    if row_lables is None:
        row_lables = [k for k, v in ratio_data]
    else:
        assert len(row_lables) == len(ratio_data), "len(row_lables) == len(ratio_data)"
    # sort based on the mean slope values
    mean_ratio = [(i,np.array(r).mean()) for i,r in enumerate(rdata)]
    mean_ratio = sorted(mean_ratio, key=lambda x:-x[1])

    indexes = [m[0] for m in mean_ratio]
    mean_ratio = np.array([m[1] for m in mean_ratio])
    pos_slope_indexes = np.array([i for i in indexes if mean_ratio[i]>0])
    neg_slope_indexes = np.array([i for i in indexes if mean_ratio[i]<0])
    std_ratio = np.array([np.array(rdata[i]).std() for i in indexes])
    row_lables = [row_lables[i] for i in indexes]
    edata = [edata[i] for i in indexes]
    mean_errors = [np.array(e).mean() for e in edata]
    fig, ax = plt.subplots(figsize=(10,6))

    medianprops = dict(linestyle='-', linewidth=2.5, color=cmaplist[4])
    err_box = ax.boxplot(edata, showfliers=False,patch_artist=True,medianprops=medianprops)
    for patch in err_box['boxes']:
        patch.set_facecolor(cmaplist[-1])
    ax.set_xticklabels(row_lables)
    ax.set_ylabel('Approximation Error')
    _ = plt.setp(ax.get_xticklabels(), rotation=-15, ha="left", va='top',
                rotation_mode="anchor")

    ax1 = ax.twinx()
    ax1.errorbar(pos_slope_indexes+1, mean_ratio[pos_slope_indexes], yerr=std_ratio[pos_slope_indexes],fmt='o',ms=10, mfc=cmaplist[1], mec=cmaplist[8], ecolor=cmaplist[2], elinewidth=3, capsize=5)
    ax1.errorbar(neg_slope_indexes+1, [0]*len(neg_slope_indexes), yerr=[0]*len(neg_slope_indexes),fmt='o',ms=10, mfc=cmaplist[0], mec=cmaplist[8], ecolor=cmaplist[2], elinewidth=3, capsize=5)

    ax1.set_ylabel('Slope')
    patches = [Line2D([0], [0], marker='o', linestyle='',color=cmaplist[1], markersize=12,mec=cmaplist[8]),
               Line2D([0], [0], color=cmaplist[2], lw=4),
               Line2D([0], [0], marker='o', linestyle='',color=cmaplist[0], markersize=12,mec=cmaplist[8]),
               Patch(color=cmaplist[-1]),
               Line2D([0], [0], color=cmaplist[4], lw=4),]
    legend = ax.legend(labels=['Slope','Slope standard deviation', 'Fail to preserve GDI','Approximation error distribution', 'Median of approximation error'], handles=patches, bbox_to_anchor=legend_pos, loc='center left', borderaxespad=0, fontsize=12, frameon=True)
    ax.grid('on')
    if filename:
        fig.savefig(filename,bbox_inches='tight')
    return row_lables, mean_ratio, mean_errors
