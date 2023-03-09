#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Any

import numpy as np
import argparse
from multiprocessing import Pool
from .utils import process_db_line, chrom_cmp_bigger, \
    prep_data, check_if_uni_sorted
from..utils import natural_chr_sort


def flexible_distance(r, q):
    """ Calculate region distance for univers """
    if r[0] <= q <= r[1]:
        return 0
    else:
        return min(abs(r[0] - q), abs(r[1] - q))


def distance(r, q):
    """ Calculate distance for hard universe"""
    return abs(r[0] - q)


def asses(db, db_que, i, current_chrom, unused_db, pos_index, flexible):
    """
    Calculate distance from given peak to the closest region in universe
    :param file db: universe file
    :param list db_que: que of three last positions in universe
    :param int i: analysed position from the query
    :param str current_chrom: current analysed chromosome from query
    :param list unused_db: list of positions from universe that were not compared to query
    :param list pos_index: which indexes from universe region use to calculate distance
    :param bool flexible: whether the universe if flexible
    :return int: peak distance to universe
    """
    if flexible:
        dist_to_db_que = [flexible_distance(j, i) for j in db_que]
    else:
        dist_to_db_que = [distance(j, i) for j in db_que]
    min_pos = np.argmin(dist_to_db_que)
    while min_pos == 2:
        d = db.readline().strip("\n")
        if d == "":
            return dist_to_db_que[min_pos]
        pos, pos_chrom = process_db_line(d, pos_index)
        if pos_chrom != current_chrom:
            unused_db.append([pos, pos_chrom])
            return dist_to_db_que[min_pos]
        db_que[:-1] = db_que[1:]
        db_que[-1] = pos
        if flexible:
            dist_to_db_que = [flexible_distance(j, i) for j in db_que]
        else:
            dist_to_db_que = [distance(j, i) for j in db_que]
        min_pos = np.argmin(dist_to_db_que)
    return dist_to_db_que[min_pos]


def process_line(db, q_chrom, current_chrom, unused_db, db_que,
                 dist, waiting, start, pos_index, flexible):
    """
    Calculate distance from new peak to universe
    :param file db: universe file
    :param str q_chrom: on which chromosome id the new peak
    :param str current_chrom: chromosome that was analysed so far
    :param list unused_db: list of positions from universe that were not compared to query
    :param list db_que: que of three last positions in universe
    :param list dist: list of all calculated distances
    :param bool waiting: whether iterating through file, without calculating
     distance,  if present chromosome not present in universe
    :param start: analysed position from the query
    :param pos_index: which indexes from universe region use to calculate distance
    :param flexible: whether the universe if flexible
    :return: if iterating through chromosome not present in universe; current chromosome in query
    """
    if q_chrom != current_chrom:
        # change chromosome
        db_que.clear()
        # clean up the que
        if len(unused_db) == 0:
            d = db.readline().strip("\n")
            if d == "":
                waiting = True
                return waiting, current_chrom
            d_start, d_start_chrom = process_db_line(d, pos_index)
            while current_chrom == d_start_chrom:
                # finish reading old chromosome in DB file
                d = db.readline().strip("\n")
                if d == "":
                    break
                d_start, d_start_chrom = process_db_line(d, pos_index)
            unused_db.append([d_start, d_start_chrom])
        current_chrom = q_chrom
        if current_chrom == unused_db[-1][1]:
            waiting = False
            db_que.append(unused_db[-1][0])
            unused_db.clear()
        elif natural_chr_sort(unused_db[-1][1], current_chrom) == 1:
            # chrom present in file not in DB
            waiting = True
            return waiting, current_chrom
        while len(db_que) < 3:
            d = db.readline().strip("\n")
            if d == "":
                break
            d_start, d_start_chrom = process_db_line(d, pos_index)
            if d_start_chrom == current_chrom:
                db_que.append(d_start)
            elif natural_chr_sort(d_start_chrom, current_chrom) == 1:
                unused_db.append([d_start, d_start_chrom])
                waiting = True
                return waiting, current_chrom
    if len(db_que) == 0:
        waiting = True
    if not waiting:
        res = asses(db, db_que, start, current_chrom, unused_db,
                    pos_index, flexible)
        dist.append(res)
    return waiting, current_chrom


def calc_distance(db_file, q_folder, q_file, flexible=False,
                  save_each=False, folder_out=None, pref=None):
    """
    For given file calculate distance to the nearst region from universe
    :param str db_file: path to universe
    :param str q_folder: path to folder containing query files
    :param str q_file: query file
    :param boolean flexible: whether the universe if flexible
    :param bool save_each: whether to save calculated distances for each file
    :param str folder_out: output folder
    :param str pref: prefix used as the name of the folder
     containing calculated distance for each file
    :return str, int, int: file name; median od distance of starts to
     starts in universe; median od distance of ends to ends in universe
    """
    prep_data(q_folder, q_file)
    q = open(os.path.join("tmp", q_file + "_sorted"), "r")
    db_start = open(db_file)
    db_que_start = []
    current_chrom_start = "chr0"
    dist_start = []
    unused_db_start = []
    waiting_start = False
    db_end = open(db_file)
    db_que_end = []
    current_chrom_end = "chr0"
    dist_end = []
    unused_db_end = []
    waiting_end = False
    pos_start = [1]
    pos_end = [2]
    if flexible:
        pos_start = [1, 6]
        pos_end = [7, 2]
    for i in q:
        i = i.split("\t")
        start = int(i[1])
        end = int(i[2])
        q_chrom = i[0]
        res_start = process_line(db_start, q_chrom, current_chrom_start,
                                 unused_db_start,
                                 db_que_start,
                                 dist_start, waiting_start, start,
                                 pos_start, flexible)
        (waiting_start, current_chrom_start) = res_start
        res_end = process_line(db_end, q_chrom, current_chrom_end,
                               unused_db_end,
                               db_que_end,
                               dist_end, waiting_end, end,
                               pos_end, flexible)
        (waiting_end, current_chrom_end) = res_end
    os.remove(os.path.join("tmp", q_file + "_sorted"))
    if save_each:
        with open(os.path.join(folder_out, pref, q_file), "w") as f:
            for i, j in zip(dist_start, dist_end):
                f.write(f"{i}\t{j}\n")
    if not dist_start:
        print(f"File {q_file} doesn't contain any chromosomes present in universe")
        return q_file, None, None
    return q_file, np.median(dist_start), np.median(dist_end)


def run_distance(folder, file_list, universe, npool, flexible,
                 save_to_file=False, folder_out=None, pref=None,
                 save_each=False):
    """
    For group of files calculate distance to the nearest region in universe
    :param str folder: path to folder containing query files
    :param str file_list: path to file containing list of query files
    :param str universe: path to universe file
    :param int npool: number of parallel processes
    :param bool flexible: whether the universe if flexible
    :param bool save_to_file: whether to save median of calculated distances for each file
    :param str folder_out: output folder
    :param str pref: prefix used for saving
    :param bool save_each: whether to save calculated distances for each file
    :return float; float: mean of median distances from starts in query to the nearest starts in universe;
    mean of median distances from ends in query to the nearest ends in universe
    """
    check_if_uni_sorted(universe)
    os.mkdir("tmp")
    files = open(file_list).read().split("\n")[:-1]
    res = []
    if folder_out:
        os.makedirs(folder_out, exist_ok=True)
    if save_each:
        os.makedirs(os.path.join(folder_out, pref))
    if npool <= 1:
        for i in files:
            r = calc_distance(universe, folder, i,
                              flexible, save_each,
                              folder_out, pref)
            res.append(r)
    else:
        with Pool(npool) as p:
            args = [(universe, folder, f, flexible,
                     save_each, folder_out, pref) for f in files]
            res = p.starmap(calc_distance, args)
    os.rmdir("tmp")
    if save_to_file:
        fout = os.path.join(folder_out, pref + "_data.tsv")
        with open(fout, "w") as o:
            o.write("file\tmedian_dist_start\tmedian_dist_end\n")
            for r in res:
                o.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
    else:
        res = np.array(res)
        res = res[:, 1:]
        res = res.astype('float')
        return np.nanmean(res[:, 0]), np.nanmean(res[:, 1])
