#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ..utils import natural_chr_sort
from .utils import check_if_uni_flexible, check_if_uni_sorted, prep_data, process_db_line


def flexible_distance_between_two_regions(region, query):
    """Calculate distance between region and flexible region from flexible universe.

    Args:
        region ([int, int]): region from flexible universe
        query (int): analyzed region

    Returns:
        int: distance
    """
    if region[0] <= query <= region[1]:
        return 0
    else:
        return min(abs(region[0] - query), abs(region[1] - query))


def distance_between_two_regions(region, query):
    """Calculate distance between region in database and region from the query.

    Args:
        region ([int]): region from hard universe
        query (int): analysed region

    Returns:
        int: distance
    """
    return abs(region[0] - query)


def distance_to_closest_region(
    db, db_queue, i, current_chrom, unused_db, pos_index, flexible, uni_to_file
):
    """Calculate distance from given peak to the closest region in database.

    Args:
        db (file): database file
        db_queue (list): queue of three last positions in database
        i: analyzed position from the query
        current_chrom (str): current analyzed chromosome from query
        unused_db (list): list of positions from universe that were not compared to query
        pos_index (list): which indexes from universe region use to calculate distance
        flexible (bool): whether the universe if flexible
        uni_to_file (bool): whether calculate distance from universe to file

    Returns:
        int: peak distance to universe
    """
    if flexible:
        if uni_to_file:
            dist_to_db_que = [flexible_distance_between_two_regions(i, j[0]) for j in db_queue]
        else:
            dist_to_db_que = [flexible_distance_between_two_regions(j, i[0]) for j in db_queue]
    else:
        dist_to_db_que = [distance_between_two_regions(j, i[0]) for j in db_queue]
    min_pos = np.argmin(dist_to_db_que)
    while min_pos == 2:
        d = db.readline().strip("\n")
        if d == "":
            return dist_to_db_que[min_pos]
        pos, pos_chrom = process_db_line(d, pos_index)
        if pos_chrom != current_chrom:
            unused_db.append([pos, pos_chrom])
            return dist_to_db_que[min_pos]
        db_queue[:-1] = db_queue[1:]
        db_queue[-1] = pos
        if flexible:
            if uni_to_file:
                dist_to_db_que = [flexible_distance_between_two_regions(i, j[0]) for j in db_queue]
            else:
                dist_to_db_que = [flexible_distance_between_two_regions(j, i[0]) for j in db_queue]
        else:
            dist_to_db_que = [distance_between_two_regions(j, i[0]) for j in db_queue]
        min_pos = np.argmin(dist_to_db_que)
    return dist_to_db_que[min_pos]


def read_in_new_universe_regions(
    db,
    q_chrom,
    current_chrom,
    unused_db,
    db_queue,
    waiting,
    pos_index,
):
    """Read in new universe regions closest to the peak.

    Args:
        db (file): universe file
        q_chrom (str): new peak's chromosome
        current_chrom (str): chromosome that was analyzed so far
        unused_db (list): list of positions from universe that were not compared to query
        db_queue (list): que of three last positions in universe
        waiting (bool): whether iterating through file, without calculating
            distance, if present chromosome not present in universe
        pos_index (list): which indexes from universe region use to calculate distance

    Returns:
        tuple: (bool, str) - if iterating through chromosome not present in universe; current chromosome in query
    """
    if q_chrom != current_chrom:
        # change chromosome
        db_queue.clear()
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
            db_queue.append(unused_db[-1][0])
            unused_db.clear()
        elif natural_chr_sort(unused_db[-1][1], current_chrom) == 1:
            # chrom present in file not in DB
            waiting = True
            return waiting, current_chrom
        while len(db_queue) < 3:
            d = db.readline().strip("\n")
            if d == "":
                break
            d_start, d_start_chrom = process_db_line(d, pos_index)
            if d_start_chrom == current_chrom:
                db_queue.append(d_start)
            elif natural_chr_sort(d_start_chrom, current_chrom) == 1:
                unused_db.append([d_start, d_start_chrom])
                waiting = True
                return waiting, current_chrom
    if len(db_queue) == 0:
        waiting = True
    return waiting, current_chrom


def calc_distance_between_two_files(
    universe,
    q_folder,
    q_file,
    flexible,
    save_each,
    folder_out,
    pref,
    uni_to_file=False,
):
    """Main function for calculating distance between regions in file query to regions in database.

    Args:
        universe (str): path to universe
        q_folder (str): path to folder containing query files
        q_file (str): query file
        flexible (bool): whether the universe if flexible
        save_each (bool): whether to save calculated distances for each file
        folder_out (str): output folder
        pref (str): prefix used as the name of the folder containing calculated distance for each file
        uni_to_file (bool): whether to calculate distance from universe to file

    Returns:
        tuple: (str, int, int) - file name; median od distance of starts to starts in universe;
            median od distance of ends to ends in universe
    """
    query = tempfile.NamedTemporaryFile()
    prep_data(q_folder, q_file, query)
    if uni_to_file:
        db_start_name = query.name
        db_end_name = query.name
        q = open(universe)
    else:
        db_start_name = universe
        db_end_name = universe
        q = query
    with open(db_start_name) as db_start, open(db_end_name) as db_end:
        db_queue_start = []
        current_chrom_start = "chr0"
        dist_start = []
        unused_db_start = []
        waiting_start = False
        db_queue_end = []
        current_chrom_end = "chr0"
        dist_end = []
        unused_db_end = []
        waiting_end = False
        start_index_q, start_index_db = [1], [1]
        end_index_q, end_index_db = [2], [2]
        if flexible and uni_to_file:
            start_index_q = [1, 6]
            end_index_q = [7, 2]
        if flexible and not uni_to_file:
            start_index_db = [1, 6]
            end_index_db = [7, 2]
        for i in q:
            if not uni_to_file:
                i = i.decode("utf-8")
            i = i.split("\t")
            start = [int(i[ind]) for ind in start_index_q]
            end = [int(i[ind]) for ind in end_index_q]
            q_chrom = i[0]
            result_start = read_in_new_universe_regions(
                db_start,
                q_chrom,
                current_chrom_start,
                unused_db_start,
                db_queue_start,
                waiting_start,
                start_index_db,
            )
            (waiting_start, current_chrom_start) = result_start
            if not waiting_start:
                result = distance_to_closest_region(
                    db_start,
                    db_queue_start,
                    start,
                    current_chrom_start,
                    unused_db_start,
                    start_index_db,
                    flexible,
                    uni_to_file,
                )
                dist_start.append(result)
            result_end = read_in_new_universe_regions(
                db_end,
                q_chrom,
                current_chrom_end,
                unused_db_end,
                db_queue_end,
                waiting_end,
                end_index_db,
            )
            (waiting_end, current_chrom_end) = result_end
            if not waiting_end:
                res = distance_to_closest_region(
                    db_end,
                    db_queue_end,
                    end,
                    current_chrom_end,
                    unused_db_end,
                    end_index_db,
                    flexible,
                    uni_to_file,
                )
                dist_end.append(res)
    query.close()
    if save_each:
        with open(os.path.join(folder_out, pref, q_file), "w") as f:
            for i, j in zip(dist_start, dist_end):
                f.write(f"{i}\t{j}\n")
    if not dist_start:
        print(f"File {q_file} doesn't contain any chromosomes present in universe")
        return q_file, None
    dist = dist_start + dist_end
    return q_file, np.median(dist)


def run_distance(
    folder,
    file_list,
    universe,
    no_workers,
    flexible=False,
    folder_out=None,
    pref=None,
    save_each=False,
    uni_to_file=False,
):
    """For group of files calculate distance to the nearest region in universe.

    Args:
        folder (str): path to folder containing query files
        file_list (str): path to file containing list of query files
        universe (str): path to universe file
        no_workers (int): number of parallel processes
        flexible (bool): whether the universe if flexible
        folder_out (str): output folder
        pref (str): prefix used for saving
        save_each (bool): whether to save calculated distances for each file
        uni_to_file (bool): whether to calculate distance from universe to file

    Returns:
        tuple: (float, float) - mean of median distances from starts in query to the nearest starts in universe;
            mean of median distances from ends in query to the nearest ends in universe
    """
    check_if_uni_sorted(universe)
    if flexible:
        check_if_uni_flexible(universe)
    with open(file_list) as f:
        files = f.read().split("\n")[:-1]
    res = []
    if save_each:
        os.makedirs(os.path.join(folder_out, pref), exist_ok=True)
    if no_workers <= 1:
        for i in files:
            r = calc_distance_between_two_files(
                universe,
                folder,
                i,
                flexible,
                save_each,
                folder_out,
                pref,
                uni_to_file,
            )
            res.append(r)
    else:
        with Pool(no_workers) as p:
            args = [
                (
                    universe,
                    folder,
                    f,
                    flexible,
                    save_each,
                    folder_out,
                    pref,
                    uni_to_file,
                )
                for f in files
            ]
            res = p.starmap(calc_distance_between_two_files, args)
    return pd.DataFrame(res)
