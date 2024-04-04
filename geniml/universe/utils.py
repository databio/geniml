#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pyBigWig


def ana_region(region, start_s):
    """Helper for saving HMM prediction into a file"""
    start_e = start_s + np.where(region == 1)[0][0]
    end_s = start_s + np.where(region == 2)[0][0]
    return start_e, end_s


def predictions_to_bed(states, chrom, bedname, save_max_cove=False, cove_file=None):
    """
    Save HMM prediction into a file
    :param ndarray states: result of HMM prediction
    :param str chrom: which chromosome is being analyzed
    :param str bedname: path to the output file
    :param bool save_max_cove: whether to save the maximum peak coverage to output
     file, can result in nonstandard bed file
    :param str cove_file: file with core coverage, require for saving maximum peak coverage
    """
    ind = np.argwhere(states != 3)
    ind = ind.flatten()
    start_s = ind[0]
    to_file = []
    line = chrom + "\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    if save_max_cove:
        coverage = pyBigWig.open(cove_file)
    for i in range(1, len(ind)):
        if ind[i] - ind[i - 1] != 1:
            end_e = ind[i - 1]
            region = states[start_s : end_e + 1]
            res = ana_region(region, start_s)
            save_start_e, save_end_s = res
            val = 0
            if save_max_cove:
                val = coverage.stats(chrom, int(start_s), int(end_e) + 1, type="max")
                val = int(val[0])
            to_file.append(
                line.format(
                    start_s,
                    end_e + 1,
                    "universe",
                    val,
                    ".",
                    save_start_e,
                    save_end_s,
                    "0,0,255",
                )
            )
            start_s = ind[i]
    if states[ind[-1]] == 2:
        region = states[start_s : ind[-1] + 1]
        res = ana_region(region, start_s)
        save_start_e, save_end_s = res
        val = 0
        if save_max_cove:
            val = coverage.stats(chrom, int(start_s), int(ind[-1]) + 1, type="max")
            val = int(val[0])
        to_file.append(
            line.format(
                start_s,
                ind[-1] + 1,
                "universe",
                val,
                ".",
                save_start_e,
                save_end_s,
                "0,0,255",
            )
        )
    with open(bedname, "a") as f:
        f.writelines(to_file)


def find_full_full_pos(seq, gap_size=1000, area_size=500):
    """Look for nonzero positions in coverage matrix, when most of the positions are zero
    :param ndarray seq: vector with information about non-zero positions
    :param int gap_size: size of minimum gap between non-zero positions that are separated
    :param int area_size: size of the area around non-zero positions to be included in the result
    :return list: list of starts of non-zero regions and list of ends of non-zero regions
    """
    size = len(seq)
    seq = np.argwhere(seq >= 1).flatten()
    starts, ends = [], []
    if seq[0] > gap_size:
        starts.append(int(seq[0] - area_size))
    else:
        starts.append(0)
    for e in range(1, len(seq)):
        if seq[e] - seq[e - 1] > gap_size:
            ends.append(int(seq[e - 1] + area_size))
            starts.append(int(seq[e] - area_size))
    ends.append(min(int(seq[-1] + area_size), size))
    return starts, ends


def find_full_empty_pos(seq, gap_size=10000, area_size=1000):
    """Look for nonzero positions in coverage matrix, when most of the positions are nonzero
    :param ndarray seq: vector with information about non-zero positions
    :param int gap_size: size of minimum gap between non-zero positions that are separated
    :param int area_size: size of the area around non-zero positions to be included in the result
    :return list: list of starts of non-zero regions and list of ends of non-zero regions
    """
    size = len(seq)
    seq = np.argwhere(seq == 0).flatten()
    starts, ends = [], []
    gap_len = 0
    gap_start = 0
    looking_for_first = True
    for e in range(1, len(seq)):
        if seq[e] - seq[e - 1] == 1:
            gap_len += 1
        else:
            if gap_len >= gap_size:
                starts.append(gap_start)
                ends.append(seq[e - 1])
                looking_for_first = False
            elif looking_for_first:
                starts.append(gap_start)
                ends.append(seq[e - 1])
                looking_for_first = False
            gap_len = 1
            gap_start = seq[e]
    starts_res = [max(0, i - area_size) for i in ends]
    end_res = [i + area_size for i in starts[1:]] + [size]
    if not starts_res:
        starts_res = [0]
    return starts_res, end_res


def find_full(seq):
    """Look for nonzero positions in coverage matrix"""
    seq = np.sum(seq, axis=1, dtype=np.uint8)
    full_pos_no = np.sum(seq >= 1)
    if full_pos_no < len(seq) - full_pos_no:
        return find_full_full_pos(seq)
    else:
        return find_full_empty_pos(seq)
