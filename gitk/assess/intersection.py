#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .utils import process_line, prep_data, check_if_uni_sorted
import os
from multiprocessing import Pool
import numpy as np
from ..utils import natural_chr_sort


def chrom_cmp(a, b):
    """Natural chromosome names comparison"""
    # com = natural_chr_sort(a, b)
    # if com < 0:
    #     return a, False, True
    # if com == 0:
    #     return a, False, False
    # if com > 0:
    #     return b, True, False
    ac = a.replace("chr", "")
    ac = ac.split("_")[0]
    bc = b.replace("chr", "")
    bc = bc.split("_")[0]
    if bc.isnumeric() and ac.isnumeric() and bc != ac:
        if int(bc) < int(ac):
            return b, True, False
        else:
            return a, False, True
    else:
        if b < a:
            return b, True, False
        else:
            return a, False, True


def relationship_helper(region_a, region_b, only_in, overlap, start_a, start_b):
    """For two region calculate their overlap; for earlier region
    calculate how many base pair only in it"""
    if region_b[0] <= region_a[1]:
        only_in += region_b[0] - region_a[0]
        if region_b[1] <= region_a[1]:
            overlap += region_b[1] - region_b[0]
            start_a, start_b = region_b[1], region_b[1]
            inside_b, inside_a = False, True
        elif region_b[1] > region_a[1]:
            overlap += region_a[1] - region_b[0]
            start_a, start_b = region_a[1], region_a[1]
            inside_a, inside_b = False, True
    elif region_b[0] > region_a[1]:
        only_in += region_a[1] - region_a[0]
        inside_a, inside_b = False, True
        start_a, start_b = region_a[1], region_b[0]
    return only_in, inside_a, inside_b, overlap, start_a, start_b


def relationship(
    region_d,
    region_q,
    only_in_d,
    only_in_q,
    inside_d,
    inside_q,
    overlap,
    start_d,
    start_q,
    waiting_d,
    waiting_q,
):
    """
    Check mutual position and calculate intersection and difference of two regions
    :param list region_d: region from universe
    :param list region_q: region from query
    :param int only_in_d: number of base pair only in universe
    :param int only_in_q: number of base pair only in query
    :param bool inside_d: whether there is still part of the region from universe to analyse
    :param bool inside_q: whether there is still part of the region from query to analyse
    :param int overlap: size of overlap
    :param int start_d: start position of currently analysed universe region
    :param int start_q: start position of currently analysed query region
    :param bool waiting_d: whether waiting for the query to finish chrom
    :param bool waiting_q: whether waiting for the universe to finish chrom
    """
    if waiting_q:
        only_in_d += region_d[1] - region_d[0]
        start_d = region_d[1]
        inside_d, inside_q = False, True
    elif waiting_d:
        only_in_q += region_q[1] - region_q[0]
        start_q = region_q[1]
        inside_q, inside_d = False, True
    else:
        if region_d[0] <= region_q[0]:
            res = relationship_helper(
                region_d, region_q, only_in_d, overlap, start_d, start_q
            )
            (only_in_d, inside_d, inside_q, overlap, start_d, start_q) = res
        if region_d[0] > region_q[0]:
            res = relationship_helper(
                region_q, region_d, only_in_q, overlap, start_q, start_d
            )
            (only_in_q, inside_q, inside_d, overlap, start_q, start_d) = res
    return only_in_d, only_in_q, inside_d, inside_q, overlap, start_d, start_q


def read_in_new_line(region, start, chrom, inside, waiting, lines, cchrom, not_e):
    if not inside:
        if not waiting:
            line = lines.readline().strip("\n")
            if line != "":
                region, start, chrom = process_line(line)
                if chrom != cchrom:
                    waiting = True
            else:
                not_e = False
                waiting = True
    return region, start, chrom, waiting, not_e


def calc_stats(db, folder, query):
    """
    Difference and overlap of two files on base pair level
    :param str db: path to universe file
    :param str folder: path to folder with query file
    :param str query: query file name
    :return str; int; int; int: file name; bp only in universe; bp only in query; overlap in bp
    """
    only_in_d, only_in_q, overlap = 0, 0, 0
    inside_d, inside_q = False, False  # inside a region
    not_end_d, not_end_q = True, True  # if there are regions to process
    waiting_d, waiting_q = False, False  # if waiting for the other file to finish chrom
    prep_data(folder, query)
    if os.stat(os.path.join("tmp", query + "_sorted")).st_size == 0:
        print(f"Empty file {query}")
        os.remove(os.path.join("tmp", query + "_sorted"))
        return [query, only_in_d, only_in_q, overlap]
    lines_q = open(os.path.join("tmp", query + "_sorted"), "r")
    lines_db = open(db)
    new_d = lines_db.readline().strip("\n")
    pos_d, start_d, chrom_d = process_line(new_d)
    new_q = lines_q.readline().strip("\n")
    pos_q, start_q, chrom_q = process_line(new_q)
    if chrom_d == chrom_q:
        c_chrom = chrom_d
    else:
        c_chrom, waiting_d, waiting_q = chrom_cmp(chrom_d, chrom_q)
    while not_end_d or not_end_q:
        res = relationship(
            [start_d, pos_d[1]],
            [start_q, pos_q[1]],
            only_in_d,
            only_in_q,
            inside_d,
            inside_q,
            overlap,
            start_d,
            start_q,
            waiting_d,
            waiting_q,
        )
        (only_in_d, only_in_q, inside_d, inside_q, overlap, start_d, start_q) = res
        new_d = read_in_new_line(
            pos_d, start_d, chrom_d, inside_d, waiting_d, lines_db, c_chrom, not_end_d
        )
        (pos_d, start_d, chrom_d, waiting_d, not_end_d) = new_d
        new_q = read_in_new_line(
            pos_q, start_q, chrom_q, inside_q, waiting_q, lines_q, c_chrom, not_end_q
        )
        (pos_q, start_q, chrom_q, waiting_q, not_end_q) = new_q
        if waiting_d or waiting_q:
            if not not_end_q:
                c_chrom = chrom_d
                waiting_d = False
            elif not not_end_d:
                c_chrom = chrom_q
                waiting_q = False
            elif chrom_d == chrom_q:
                c_chrom = chrom_d
                waiting_d, waiting_q = False, False
            elif waiting_d and waiting_q:
                c_chrom, waiting_d, waiting_q = chrom_cmp(chrom_d, chrom_q)

    only_in_d += pos_d[1] - start_d
    only_in_q += pos_q[1] - start_q
    os.remove(os.path.join("tmp", query + "_sorted"))
    return [query, only_in_d, only_in_q, overlap]


def run_intersection(
    folder, file_list, universe, npool, save_to_file=False, folder_out=None, pref=None
):
    """
    Calculate the base pair intersection of universe and group of files
    :param str folder: path to folder containing query files
    :param str file_list: path to file containing list of query files
    :param str universe: path to universe file
    :param int npool: number of parallel processes
    :param str save_to_file: whether to save median of calculated distances for each file
    :param str folder_out: output folder
    :param str pref: prefix used for saving
    :return float; float: mean of fractions of intersection of file and universe divided by universe size;
    mean of fractions of intersection of file and universe divided by file size
    """
    check_if_uni_sorted(universe)
    if save_to_file:
        os.makedirs(folder_out, exist_ok=True)
    os.mkdir("tmp")
    files = open(file_list).read().split("\n")[:-1]
    res = []
    if npool <= 1:
        for i in files:
            r = calc_stats(universe, folder, i)
            res.append(r)
    else:
        with Pool(npool) as p:
            args = [(universe, folder, f) for f in files]
            res = p.starmap(calc_stats, args)
    os.rmdir("tmp")
    if save_to_file:
        fout = os.path.join(folder_out, pref + "_data.tsv")
        with open(fout, "w") as o:
            o.write("file\tunivers/file\tfile/universe\tuniverse&file\n")
            for r in res:
                o.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")
    else:
        res = np.array(res)
        res = res[:, 1:]
        res = res.astype("float")
        recall = res[:, 2] / (res[:, 2] + res[:, 1])
        precision = res[:, 2] / (res[:, 2] + res[:, 0])
        F_10 = (1 + 10**2) * (precision * recall) / ((10**2 * precision) + recall)
        return np.median(F_10)
