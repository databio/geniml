import numpy as np
import os
from .models import PoissonModel
import pyBigWig
from scipy.stats import nbinom
from functools import cmp_to_key
from..utils import natural_chr_sort

from logging import getLogger
from ..const import PKG_NAME

_LOGGER = getLogger(PKG_NAME)

transmat = [[1 - 1e-10, 0, 0, 1e-10],
            [0, 1 - 1e-6, 1e-6, 0],
            [0.1, 0, 0.9, 0],
            [0, 1e-6, 0, 1 - 1e-6]]

lambdas = [[3, 0.0001, 1],
           [0.0001, 3, 1],
           [1e-4, 1e-4, 1e-3],
           [0.05, 0.05, 2]]

WINDOW_SIZE = 26
FIX_UNIWIG = False


def norm(track, mode):
    """ Normalize the coverage track depending on track type.
    For each unique value in the track calculates the corresponding
    quantile taking into account that values occur different number of times. """
    important_val = track[track != 0]
    important_val_unique, counts = np.unique(important_val,
                                             return_counts=True)
    uniq_dict = {i: j for i, j in zip(important_val_unique, counts)}
    # how many times each value is present in the track
    important_val_unique_sort = np.sort(important_val_unique)
    if mode == "ends":
        n = 0.1
    if mode == "core":
        n = 0.085
    bs = 0  # what fraction of the distribution was used for normalization
    val = {}  # for each unique value in track holds the corresponding quantile
    for i in important_val_unique_sort:
        move_val = (uniq_dict[i] / len(important_val)) / 2
        # how far from last quantile is te next one
        val[i] = nbinom.ppf(bs + move_val, 1, n)
        bs = bs + move_val * 2
    track[track != 0] = [val[i] for i in important_val]


def process_bigwig(file, seq, p, chrom, chrom_size,
                   normalize=False, mode=None, fix_uniwig=False):
    """ Preprocess bigWig file """
    if pyBigWig.numpy:
        track = file.values(chrom, 0, chrom_size, numpy=True)
    else:
        track = file.values(chrom, 0, chrom_size)
        track = np.array(track)
    track[np.isnan(track)] = 0
    track = track.astype(np.uint16)
    if fix_uniwig and mode != "core":
        track = np.pad(track[WINDOW_SIZE:], (0, WINDOW_SIZE))
    if normalize:
        norm(track, mode)
    seq[:, p] = track


def read_data(start, end, cove, chrom,
              normalize=False):
    """
    Read in and preprocess data
    :param str start: path to file with start coverage
    :param str end: path to file with end coverage
    :param str cove: path to file with  core coverage
    :param str chrom: chromosome to analyse
    :param boolean normalize: whether to normalize the coverage
    :return: chromosome size, coverage matrix
    """
    start = pyBigWig.open(start + ".bw")
    chroms = start.chroms()
    chrom_size = chroms[chrom]
    seq = np.zeros((chrom_size, 3), dtype=np.uint16)
    process_bigwig(start, seq, 0, chrom, chrom_size,
                   normalize, mode="ends", fix_uniwig=FIX_UNIWIG)
    start.close()
    end = pyBigWig.open(end + ".bw")
    process_bigwig(end, seq, 1, chrom, chrom_size,
                   normalize, mode="ends", fix_uniwig=FIX_UNIWIG)
    end.close()
    cove = pyBigWig.open(cove + ".bw")
    process_bigwig(cove, seq, 2, chrom, chrom_size,
                   normalize, mode="core")
    cove.close()
    return chrom_size, seq


def find_full_full_pos(seq, gap_size=1000, area_size=500):
    """ Look for nonzero positions in coverage matrix,
     when most of the positions are zero """
    size = len(seq)
    seq = np.argwhere(seq >= 1).flatten()
    starts, ends = [], []
    if seq[0] > gap_size:
        starts.append(int(seq[0] - area_size))
    else:
        starts.append(1)
    for e in range(1, len(seq)):
        if seq[e] - seq[e - 1] > gap_size:
            ends.append(int(seq[e - 1] + area_size))
            starts.append(int(seq[e] - area_size))
    ends.append(min(int(seq[-1] + area_size), size))
    return starts, ends


def find_full_empty_pos(seq, gap_size=10000, area_size=1000):
    """ Look for nonzero positions in coverage matrix,
     when most of the positions are nonzero """
    size = len(seq)
    seq = np.argwhere(seq == 0).flatten()
    starts, ends = [], []
    gap_len = 0
    gap_start = 0
    looking_for_first = True
    for e in range(1, len(seq)):
        if seq[e] - seq[e-1] == 1:
            gap_len += 1
        else:
            if gap_len >= gap_size:
                starts.append(gap_start)
                ends.append(seq[e-1])
                looking_for_first = False
            elif looking_for_first:
                starts.append(gap_start)
                ends.append(seq[e - 1])
                looking_for_first = False
            gap_len = 1
            gap_start = seq[e]
    starts_res = [i-area_size for i in ends]
    end_res = [i+area_size for i in starts[1:]] + [size]
    if not starts_res:
        starts_res = [0]
    return starts_res, end_res


def find_full(seq):
    """ Look for nonzero positions in coverage matrix """
    seq = np.where(seq > 0, 1, 0).astype(np.uint8)
    seq = np.sum(seq, axis=1, dtype=np.uint8)
    full_pos_no = np.sum(seq >= 1)
    if full_pos_no < len(seq) - full_pos_no:
        return find_full_full_pos(seq)
    else:
        return find_full_empty_pos(seq)


def ana_region(region, start_s):
    """ Helper for saving HMM prediction into a file """
    start_e = start_s + np.where(region == 3)[0][0]
    end_s = start_s + np.where(region == 1)[0][0]
    return start_e, end_s


def hmm_pred_to_bed(states, chrom, bedname, save_max_cove=False, cove_file=None):
    """
    Save HMM prediction into a file
    :param array states: result of HMM prediction
    :param str chrom: which chromosome is being analysed
    :param str bedname: path to the output file
    :param boolean save_max_cove: whether to save the maximum peak
     coverage to output file, can result in nonstandard bed file
    :param str cove_file: file with core coverage, require for
     saving maximum peak coverage
    """
    ind = np.argwhere(states != 2)
    ind = ind.flatten()
    start_s = ind[0]
    to_file = []
    line = chrom + "\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    if save_max_cove:
        coverage = pyBigWig.open(cove_file)
    for i in range(1, len(ind)):
        if ind[i] - ind[i - 1] != 1:
            end_e = ind[i - 1]
            region = states[start_s:end_e + 1]
            res = ana_region(region, start_s)
            save_start_e, save_end_s = res
            val = 0
            if save_max_cove:
                val = coverage.stats(chrom, int(start_s), int(end_e) + 1, type="max")
                val = int(val[0])
            to_file.append(line.format(start_s, end_e + 1,
                                       'universe', val, '.',
                                       save_start_e, save_end_s, '0,0,255'))
            start_s = ind[i]
    if states[ind[-1]] == 1:
        region = states[start_s:ind[-1] + 1]
        res = ana_region(region, start_s)
        save_start_e, save_end_s = res
        val = 0
        if save_max_cove:
            val = coverage.stats(chrom, int(start_s), int(ind[-1]) + 1, type="max")
            val = int(val[0])
        to_file.append(line.format(start_s, ind[-1] + 1,
                                   'universe', val, '.',
                                   save_start_e, save_end_s, '0,0,255'))
    with open(bedname, "a") as f:
        f.writelines(to_file)


def split_predict(seq, empty_starts, empty_ends, model):
    """ Make model prediction only for regions containing
     nonzero positions  """
    hmm_predictions = np.full(len(seq), 2, dtype=np.uint8)
    for s, e in zip(empty_starts, empty_ends):
        res = model.predict(seq[s:e])
        hmm_predictions[s:e] = res
    return hmm_predictions


def run_hmm(start, end, cove, chrom, normalize=False):
    """ Make HMM prediction for given chromosome """
    chrom_size, seq = read_data(start, end, cove,
                                chrom,
                                normalize=normalize)
    empty_starts, empty_ends = find_full(seq)
    model = PoissonModel(transmat, lambdas, save_matrix=False)
    model = model.make()
    hmm_predictions = split_predict(seq, empty_starts, empty_ends, model)
    return hmm_predictions, model


def run_hmm_save_bed(start, end, cove, out_file, normalize, save_max_cove):
    """
    Create HMM based univers from coverage
    :param str start: path to start coverage file
    :param str end: path to end coverage file
    :param str cove: path to core coverage file
    :param str out_file: path to the output file with universe
    :param boolean normalize: whether to normalize file
    :param boolean save_max_cove: whether to save the maximum
    peak coverage
    """
    if os.path.isfile(out_file):
        raise Exception(f"File : {out_file} exists")
    bw_start = pyBigWig.open(start + ".bw")
    chroms = bw_start.chroms()
    bw_start.close()
    chroms_key = list(chroms.keys())
    chroms_key = sorted(chroms_key, key=cmp_to_key(natural_chr_sort))
    chroms = {i: chroms[i] for i in chroms_key}
    chroms = {"chr21": chroms["chr21"]}
    for C in chroms:
        if chroms[C] > 0:
            pred, m = run_hmm(start, end, cove, C, normalize=normalize)
            hmm_pred_to_bed(
                pred, C, out_file, save_max_cove=save_max_cove, cove_file=cove + ".bw"
            )


def test_hmm(message):
    """ Just prints a test message """
    _LOGGER.info(message)
