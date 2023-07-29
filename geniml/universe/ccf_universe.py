#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from functools import cmp_to_key

import numpy as np
import pyBigWig

from geniml.utils import natural_chr_sort, timer_func


def ana_region(reg, start_s, starts, ends, track_val):
    """Check how many regions given part of universe contains
    :param ndarray reg: vector with universes states; 0 - background, 1- boundary, 2- core
    :param int start_s: start of the part of the universe
    :param list starts: list of region starts in given part of universe
    :param list ends: list of region ends in given part of universe
    :param ndarray track_val: genome coverage by the collection for given part of universe
    """
    core_s, core_e = [], []
    core_pos = np.argwhere(reg == 2).flatten()
    if len(core_pos) == 0:
        return "empty"
    else:
        core_s.append(core_pos[0] + start_s)
        if core_pos[0] == 0:
            core_s[0] += 1
        for i in range(1, len(core_pos)):
            if core_pos[i] - core_pos[i - 1] >= 50:
                core_e.append(core_pos[i - 1] + start_s)
                min_point = np.argmin(track_val[core_pos[i - 1] : core_pos[i]])
                min_point = min_point + core_pos[i - 1]
                ends.append(int(min_point) + start_s)
                starts.append(ends[-1] + 1)
                core_s.append(core_pos[i] + start_s)
                if core_s[-1] == starts[-1]:
                    core_s[-1] += 1
        core_e.append(core_pos[-1] + start_s)
        if core_pos[-1] == len(reg) - 1:
            core_e[-1] -= 1
    return core_s, core_e, starts, ends


def save_regions(inter_pos, chrom, bedname, track):
    """Save regions from universes to file
    :param ndarray inter_pos: vector with universes states; 0 - background, 1- boundary, 2- core
    :param str chrom: chromosome to analyse
    :param str bedname: output file
    :param ndarray track: vector with coverage values
    """
    ind = np.argwhere(inter_pos != 0)
    ind = ind.flatten()
    start_s = ind[0]
    to_file = []
    line = chrom + "\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"
    for i in range(1, len(ind)):
        if ind[i] - ind[i - 1] != 1:
            end_e = ind[i - 1]
            region = inter_pos[start_s : end_e + 1]
            res = ana_region(region, start_s, [start_s], [], track[start_s : end_e + 1])
            if res != "empty":
                save_start_e, save_end_s, save_start_s, save_end_e = res
                save_end_e = save_end_e + [end_e]
                for a, b, c, d in zip(save_start_e, save_end_s, save_start_s, save_end_e):
                    if a != b:
                        val = 0
                        li = line.format(
                            int(c),
                            int(d) + 1,
                            "universe",
                            val,
                            ".",
                            int(a),
                            int(b),
                            "0,0,255",
                        )
                        to_file.append(li)
            start_s = ind[i]
    end_e = ind[-1]
    region = inter_pos[start_s : end_e + 1]
    res = ana_region(region, start_s, [start_s], [], track[start_s : end_e + 1])
    if res != "empty":
        save_start_e, save_end_s, save_start_s, save_end_e = res
        save_end_e = save_end_e + [end_e]
        for a, b, c, d in zip(save_start_e, save_end_s, save_start_s, save_end_e):
            val = 0
            li = line.format(
                int(c),
                int(d) + 1,
                "universe",
                val,
                ".",
                int(a),
                int(b),
                "0,0,255",
            )
            to_file.append(li)
    with open(bedname, "a") as f:
        f.writelines(to_file)


def get_uni(file, chrom, bedname):
    """Build cut-off coverage flexible universes from coverage track
    :param str file: coverage file
    :param str chrom: chromosome to analyse
    :param str bedname: output file
    """
    file = pyBigWig.open(file)
    if pyBigWig.numpy:
        track = file.values(chrom, 0, file.chroms(chrom), numpy=True)
    else:
        track = file.values(chrom, 0, file.chroms(chrom))
        track = np.array(track)
    track[np.isnan(track)] = 0
    track = track.astype(np.uint16)
    cutoff = np.sum(track) / len(track)
    track_non_zero_sort = np.sort(track[track != 0])

    cutoff = max([1, np.round(cutoff)])
    pos = np.where(track_non_zero_sort == cutoff)[0]
    if len(pos) == 0:
        uniq_val = np.unique(track_non_zero_sort)
        dist = np.absolute(uniq_val - cutoff)
        cutoff = uniq_val[dist.argmin()]
        pos = np.where(track_non_zero_sort == cutoff)[0]
    f, l = pos[0] / len(track_non_zero_sort), pos[-1] / len(track_non_zero_sort)
    q_cutoff = np.mean([f, l])
    lower = np.quantile(track_non_zero_sort, max(0, q_cutoff - 0.2))
    upper = np.quantile(track_non_zero_sort, min(1, q_cutoff + 0.2))
    inter_pos = np.zeros(len(track), dtype=np.uint8)
    inter_pos[track >= lower] = 1
    inter_pos[track > upper] = 2
    save_regions(inter_pos, chrom, bedname, track)


def ccf_universe(cove, file_out, cove_prefix="all"):
    """
    Creat cut-off flexible universe based on coverage track
    :param str cove: path to coverage folder
    :param str file_out: output file
    :param str cove_prefix: prefixed used for creating signal tracks
    """
    if os.path.isfile(file_out):
        raise Exception(f"File : {file_out} exists")
    file = os.path.join(cove, f"{cove_prefix}_core.bw")
    bw = pyBigWig.open(file)
    chroms = bw.chroms()
    bw.close()
    chroms_key = list(chroms.keys())
    chroms_key = sorted(chroms_key, key=cmp_to_key(natural_chr_sort))
    chroms = {i: chroms[i] for i in chroms_key}
    for chrom in chroms:
        if chroms[chrom] > 0:
            get_uni(file, chrom, file_out)
