import os
import warnings
from logging import getLogger

import numpy as np
import pandas as pd

from ..const import PKG_NAME
from .distance import run_distance
from .intersection import run_intersection
from .likelihood import hard_universe_likelihood, likelihood_flexible_universe
from .utils import check_if_uni_flexible

_LOGGER = getLogger(PKG_NAME)


def run_all_assessment_methods(
    raw_data_folder,
    file_list,
    universe,
    no_workers,
    folder_out,
    pref,
    save_each,
    overlap=False,
    distance_f_t_u=False,
    distance_f_t_u_flex=False,
    distance_u_t_f=False,
    distance_u_t_f_flex=False,
):
    """
    Assess universe fit to collection using overlap and distance metrics
    :param str raw_data_folder: path to raw files from the collection
    :param str file_list: path to file with list of files in the collection
    :param str universe: path to universe that is being assessed
    :param int no_workers: number of workers for multiprocessing
    :param str folder_out: output folder
    :param str pref: prefixed used for creating output files
    :param bool save_each: if save output of distance metrics for each region
    :param bool overlap: if calculate overlap metrics
    :param bool distance_f_t_u: if calculate distance from file to universe metrics
    :param bool distance_f_t_u_flex: if calculate flexible distance from file to universe metrics
    :param bool distance_u_t_f: if calculate distance from universes to file metrics
    :param bool distance_u_t_f_flex: if calculate flexible distance from universes to file metrics
    """
    if not any(
        [
            overlap,
            distance_f_t_u,
            distance_f_t_u_flex,
            distance_u_t_f,
            distance_u_t_f_flex,
        ]
    ):
        raise AttributeError("Choose at least one assessment method")
    if not any(
        [
            distance_f_t_u,
            distance_f_t_u_flex,
            distance_u_t_f,
            distance_u_t_f_flex,
        ]
    ):
        warnings.warn("Unused argument: save_each")
    asses_results = []
    if overlap:
        r_overlap = run_intersection(
            raw_data_folder,
            file_list,
            universe,
            no_workers,
        )
        r_overlap.columns = [
            "file",
            "univers/file",
            "file/universe",
            "universe&file",
        ]
        asses_results.append(r_overlap)
        _LOGGER.info("DONE: Overlap")
    if distance_f_t_u:
        r_distance = run_distance(
            raw_data_folder,
            file_list,
            universe,
            no_workers,
            False,
            folder_out,
            pref + "_dist_file_to_universe",
            save_each,
            False,
        )
        r_distance.columns = ["file", "median_dist_file_to_universe"]
        asses_results.append(r_distance)
        _LOGGER.info("DONE: Distance file to universe")
    if distance_f_t_u_flex:
        r_distance_flex = run_distance(
            raw_data_folder,
            file_list,
            universe,
            no_workers,
            True,
            folder_out,
            pref + "_dist_file_to_universe_flex",
            save_each,
            False,
        )
        r_distance_flex.columns = ["file", "median_dist_file_to_universe_flex"]
        asses_results.append(r_distance_flex)
        _LOGGER.info("DONE: Flexible distance file to universe")

    if distance_u_t_f:
        r_distance_utf = run_distance(
            raw_data_folder,
            file_list,
            universe,
            no_workers,
            False,
            folder_out,
            pref + "_dist_universe_to_file",
            save_each,
            True,
        )
        r_distance_utf.columns = ["file", "median_dist_universe_to_file"]
        asses_results.append(r_distance_utf)
        _LOGGER.info("DONE: Distance universe to file")
    if distance_u_t_f_flex:
        r_distance_utf_flex = run_distance(
            raw_data_folder,
            file_list,
            universe,
            no_workers,
            True,
            folder_out,
            pref + "median_dist_universe_to_file_flex",
            save_each,
            True,
        )
        r_distance_utf_flex.columns = [
            "file",
            "median_dist_universe_to_file_flex",
        ]
        asses_results.append(r_distance_utf_flex)
        _LOGGER.info("DONE: Flexible distance universe to file")

    df = asses_results[0]
    for i in asses_results[1:]:
        df = pd.merge(df, i, on="file")
    df.to_csv(os.path.join(folder_out, pref + "_data.csv"), index=False)


def get_rbs(f_t_u, u_t_f):
    """
    Calculate RBS
    """
    a = 101 / (f_t_u + 100)
    b = 101 / (u_t_f + 100)
    rbs = (10 * a + b) / 11
    return rbs


def get_mean_rbs(folder, file_list, universe, no_workers, flexible=False):
    """
    Calculate average RBS of the collection
    :param str folder: path to folder with the collection
    :param str file_list: path to file with list of files in the collection
    :param str universe: path to the universe
    :param int no_workers: number of workers for multiprocessing
    :param bool flexible: if to calculate flexible version of the metric
    :return int: average RBS
    """
    file_to_uni = run_distance(
        folder,
        file_list,
        universe,
        no_workers,
        flexible=flexible,
        uni_to_file=False,
    )

    uni_to_file = run_distance(
        folder,
        file_list,
        universe,
        no_workers,
        flexible=flexible,
        uni_to_file=True,
    )
    rbs = get_rbs(file_to_uni[1], uni_to_file[1])
    return np.mean(rbs)


def get_rbs_from_assessment_file(file, cs_each_file=False, flexible=False):
    """
    Calculate RBS form file with results of metrics per file
    :param str file: path to file with assessment results
    :param bool cs_each_file: if report RBS for each file, not average for the collection
    :param bool flexible: if use flexible version of the metric
    """
    df = pd.read_csv(file, index_col=(0))
    if flexible:
        df["f_t_u"] = df["median_dist_file_to_universe_flex"]
        df["u_t_f"] = df["median_dist_universe_to_file_flex"]
    else:
        df["f_t_u"] = df["median_dist_file_to_universe"]
        df["u_t_f"] = df["median_dist_universe_to_file"]
    df["RBS"] = get_rbs(df["f_t_u"], df["u_t_f"])
    if cs_each_file:
        return df
    else:
        return df["RBS"].mean()


def get_f_10_score(
    folder,
    file_list,
    universe,
    no_workers,
):
    """
    Get F10 score for a universes and collection of files
    :param str folder: path to folder with the collection
    :param str file_list: path to file with list of files in the collection
    :param str universe: path to the universe
    :param int no_workers: number of workers for multiprocessing
    :return int: average F10 score
    """
    res = run_intersection(
        folder,
        file_list,
        universe,
        no_workers,
    )
    res = np.array(res)
    res = res[:, 1:]
    res = res.astype("float")
    recall = res[:, 2] / (res[:, 2] + res[:, 1])
    precision = res[:, 2] / (res[:, 2] + res[:, 0])
    f_10 = (1 + 10**2) * (precision * recall) / ((10**2 * precision) + recall)
    return np.mean(f_10)


def get_f_10_score_from_assessment_file(file, f10_each_file=False):
    """
    Get F10 score from assessment output file
    :param str file: path to file with assessment results
    :param bool f10_each_file: if report F10 for each file, not average for the collection
    """
    df = pd.read_csv(file, index_col=(0))
    r = df["A&U/A"]
    p = df["A&U/U"]
    df["F_10"] = (1 + 10**2) * (p * r) / ((10**2 * p) + r)
    if f10_each_file:
        return df["F_10"]
    else:
        return df["F_10"].mean()


def get_likelihood(
    model_file,
    universe,
    cove_folder,
    cove_prefix="all",
    flexible=False,
    save_peak_input=False,
):
    """
    Calculate universe likelihood given collection
    :param str model_file: path to file with likelihood model
    :param str universe: path to the universe
    :param str cove_folder: path to the coverage folder
    :param str cove_prefix: prefixed used for generating coverage
    :param bool flexible: if to calculate flexible likelihood
    :param bool save_peak_input:  if to save likelihood input of each region
    :return:
    """
    if flexible:
        lh = likelihood_flexible_universe(
            model_file, universe, cove_folder, cove_prefix, save_peak_input
        )
    else:
        if save_peak_input:
            warnings.warn("Unused argument: save_peak_input")
        lh = hard_universe_likelihood(model_file, universe, cove_folder, cove_prefix)

    return lh


def filter_universe(
    universe,
    universe_filtered,
    min_size=0,
    min_coverage=0,
    filter_lh=False,
    model_file=None,
    cove_folder=None,
    cove_prefix=None,
    lh_cutoff=0,
):
    """
    Filter universe by region size, coverage by collection, likelihood
    :param str universe: path to input universe
    :param str universe_filtered: path to output filtered universe
    :param int min_size: minimum size of the region in the output universe
    :param int min_coverage: minimum number coverage of universe region by collection
    :param bool filter_lh: if use likelihood to filter universe
    :param str model_file: path to collection likelihood model
    :param str cove_folder: path to folder with coverage tracks
    :param str cove_prefix: prefixed used for creating tracks
    :param int lh_cutoff: minimum likelihood input
    """
    if filter_lh:
        check_if_uni_flexible(universe)
        if not all([model_file, cove_folder, cove_prefix]):
            miss_args = []
            if not model_file:
                miss_args.append("model_file")
            if not cove_folder:
                miss_args.append("cove_folder")
            if not cove_prefix:
                miss_args.append("cove_prefix")
            raise ValueError(
                "Missing {} for peak likelihood calculations.".format(",".join(miss_args))
            )
        likelihood_flexible_universe(model_file, universe, cove_folder, cove_prefix, True)
        universe = universe + "_peak_likelihood"
    with open(universe) as uni:
        with open(universe_filtered, "w+") as uni_flt:
            for i in uni:
                j = i.split("\t")
                j[1], j[2], j[4] = int(j[1]), int(j[2]), int(j[4])
                if j[2] - j[1] > min_size:
                    if j[4] > min_coverage:
                        if filter_lh:
                            if float(j[9].strip("\n")) > lh_cutoff:
                                uni_flt.write(i)
                        else:
                            uni_flt.write(i)
