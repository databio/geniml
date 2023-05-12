#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from .utils import chrom_cmp_bigger, process_db_line, prep_data, check_if_uni_sorted
import numpy as np
from multiprocessing import Pool


def process_region(
    start, start_region, q_chrom, start_region_chrom, db, b_index, e_index
):
    """
    For given position check if it is covered by universe flexible region
    :param int start: position to analyse from query peak
    :param list start_region: last analysed region from universe
    :param str q_chrom: chromosome of query peak
    :param str start_region_chrom:  chromosome of universe region
    :param file db: universe file
    :param int b_index: index of region beginning in line from universe
    :param int e_index: index of region ending in line from universe
    :return: whether peak is covered; current region from universe; current chromosome of region from universe
    """
    while chrom_cmp_bigger(q_chrom, start_region_chrom):
        dn = db.readline().strip("\n")
        if dn == "":
            break
        start_region, start_region_chrom = process_db_line(dn, [b_index, e_index])
    while start > start_region[1] and q_chrom == start_region_chrom:
        dn = db.readline().strip("\n")
        if dn == "":
            break
        start_region, start_region_chrom = process_db_line(dn, [b_index, e_index])
    if start_region[0] <= start < start_region[1] and q_chrom == start_region_chrom:
        return True, start_region, start_region_chrom
    return False, start_region, start_region_chrom


def calc_no_retrieve(db_file, q_folder, q_file):
    """
    Calculate percent of strats and ends covered by flexible universe for given file
    :param str db_file: path to universe file
    :param str q_folder: path to folder containing query files
    :param str q_file: file name
    :return: file name; number of peaks in file; number of peaks with start covered by universe;
    percent of peaks with start covered by universe; number of peaks with end covered by universe;
    percent of peaks with end covered by universe; number of peaks with at least on end covered by universe;
    percent of peaks with at least on end covered by universe; number of peaks with both ends covered by universe;
    percent of peaks with both ends covered by universe;
    """
    prep_data(q_folder, q_file)
    q = open(os.path.join("tmp", q_file + "_sorted"), "r")
    db_start = open(db_file)
    d_start = db_start.readline().strip("\n")
    start_region, start_region_chrom = process_db_line(d_start, [1, 6])
    db_end = open(db_file)
    d_end = db_end.readline().strip("\n")
    end_region, end_region_chrom = process_db_line(d_end, [1, 6])
    res_start = 0
    res_end = 0
    res_or = 0
    res_and = 0
    q_len = 0
    for i in q:
        q_len += 1
        i = i.split("\t")
        start = int(i[1])
        end = int(i[2])
        q_chrom = i[0]
        start_out = process_region(
            start, start_region, q_chrom, start_region_chrom, db_start, 1, 6
        )
        (found_start, start_region, start_region_chrom) = start_out
        end_out = process_region(
            end, end_region, q_chrom, end_region_chrom, db_end, 7, 2
        )
        (found_end, end_region, end_region_chrom) = end_out
        if found_start:
            res_start += 1
        if found_end:
            res_end += 1
        if found_start or found_end:
            res_or += 1
        if found_start and found_end:
            res_and += 1
    os.remove(os.path.join("tmp", q_file + "_sorted"))
    return (
        q_file,
        q_len,
        res_start,
        res_start / q_len * 100,
        res_end,
        res_end / q_len * 100,
        res_or,
        res_or / q_len * 100,
        res_and,
        res_and / q_len * 100,
    )


def run_recovered(
    q_folder, file_list, db_file, npool, save_to_file=False, folder_out=None, pref=None
):
    """
    Calculate percent of strats and ends covered by flexible universe for set of files
    :param str q_folder: path to folder containing query files
    :param str file_list: path to file containing list of query files
    :param str db_file: path to universe file
    :param int npool: number of parallel processes
    :param bool save_to_file: whether to save median of calculated distances for each file
    :param str folder_out: output folder
    :param str pref: prefix used for saving
    :return: mean of percent of strats covered by flexible universe;
    mean of percent of ends covered by flexible universe;
    mean of percent of regions with at least one end covered by flexible universe;
    mean of percent of regions with both ends covered by flexible universe
    """
    check_if_uni_sorted(db_file)
    if folder_out:
        os.makedirs(folder_out, exist_ok=True)
    os.mkdir("tmp")
    files = open(file_list).read().split("\n")[:-1]
    res = []
    if npool <= 1:
        for i in files:
            r = calc_no_retrieve(db_file, q_folder, i)
            res.append(r)
    else:
        with Pool(npool) as p:
            args = [(db_file, q_folder, f) for f in files]
            res = p.starmap(calc_no_retrieve, args)
    os.rmdir("tmp")
    if save_to_file:
        fout = os.path.join(folder_out, pref + "_data.tsv")
        with open(fout, "w") as o:
            header = [
                "file",
                "peak_no",
                "peak_start_percent",
                "peak_end_percent",
                "peak_or_percent",
                "peak_and_percent",
            ]
            o.write("\t".join(header) + "\n")
            for r in res:
                o.write(f"{r[0]}\t{r[1]}\t{r[3]}\t{r[5]}\t{r[7]}\t{r[9]}\n")
    else:
        res = np.array(res)
        res = res[:, [1, 3, 5, 7, 9]]
        res = res.astype("float")
        return np.mean(res, axis=0)
