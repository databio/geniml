#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from time import time
from functools import cmp_to_key
import pyBigWig


def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1)/60:.4f}min')
        return result
    return wrap_func


def natural_chr_sort(a, b):
    """ Compare chromosomes based on natural order"""
    ac = a.replace("chr", "")
    ac = ac.split("_")[0]
    bc = b.replace("chr", "")
    bc = bc.split("_")[0]
    if bc.isnumeric() and ac.isnumeric() and bc != ac:
        if int(bc) < int(ac):
            return 1
        elif int(bc) > int(ac):
            return -1
        else:
            return 0
    else:
        if b < a:
            return 1
        elif a < b:
            return -1
        else:
            return 0


def get_uni(file, chrom, cut_off=None):
    """For each position check if coverage is bigger than cut-off;
    if cut-off not provided calculate value that gives
    maximum likelihood universe
    :param str file: coverage file
    :param str chrom: chromosome to analyse
    :param int cut_off: base pairs with values grater of equal to cut-off can be included in universe
    :return: """
    file = pyBigWig.open(file)
    if pyBigWig.numpy:
        track = file.values(chrom, 0, file.chroms(chrom), numpy=True)
    else:
        track = file.values(chrom, 0, file.chroms(chrom))
        track = np.array(track)
    track[np.isnan(track)] = 0
    track = track.astype(np.uint16)
    if cut_off is None:
        cut_off = np.sum(track)/len(track)
    inter_pos = track >= cut_off
    file.close()
    return inter_pos


def save_simple(fout, inter_pos, chrom):
    """
    Save cut-off universe to a file without any processing
    :param str fout: output file
    :param bool vector inter_pos: whether each position should be included in universe
    :param str chrom: chromosome to analyse
    """
    inter_pos_uni = np.argwhere(inter_pos)
    start = inter_pos_uni[0][0]
    with open(fout, "a") as f:
        for i in range(1, len(inter_pos_uni)):
            if inter_pos_uni[i] - inter_pos_uni[i - 1] != 1:
                end = inter_pos_uni[i - 1][0]
                f.write(f"{chrom}\t{start}\t{end}\n")
                start = inter_pos_uni[i][0]


def marge_filter(fout, inter_pos, chrom, merge_dist=100, size_flt=1000):
    """
    Save cut-off universe to a file with filtering region size and merging close regions
    :param fout: output file
    :param bool vector inter_pos: whether each position should be included in universe
    :param str chrom: chromosome to analyse
    :param int merge_dist: regions closer than merge_dist will be merged into one
    :param int size_flt: regions smaller than size_flt will not be reported
    """
    inter_pos_uni = np.argwhere(inter_pos)
    start = inter_pos_uni[0][0]
    with open(fout, "a") as f:
        for i in range(1, len(inter_pos_uni)):
            if inter_pos_uni[i] - inter_pos_uni[i - 1] >= merge_dist:
                end = inter_pos_uni[i - 1][0]
                if end - start >= size_flt:
                    f.write(f"{chrom}\t{start}\t{end}\n")
                start = inter_pos_uni[i][0]


@timer_func
def main(file, fout, merge=0, filter_size=0,
         cut_off=None):
    """
    Creat cut-off universe based on coverage track
    :param str file: path to coverage file without extension
    :param int merge: regions closer than this value will be merged into one
    :param int filter_size: regions smaller than this value will not be reported
    :param str fout: output file
    :param int cut_off: base pairs with coverage equal to or greater than this value will be included in the universe
    """
    if os.path.isfile(fout):
        raise Exception(f"File : {fout} exists")
    bw_start = pyBigWig.open(file)
    chroms = bw_start.chroms()
    bw_start.close()
    chroms_key = list(chroms.keys())
    chroms_key = sorted(chroms_key, key=cmp_to_key(natural_chr_sort))
    chroms = {i: chroms[i] for i in chroms_key}
    for chrom in chroms:
        inter_pos = get_uni(file, chrom, cut_off)
        if merge == 0 and filter_size == 0:
            save_simple(fout, inter_pos, chrom)
        else:
            marge_filter(fout, inter_pos, chrom,
                         merge, filter_size)
