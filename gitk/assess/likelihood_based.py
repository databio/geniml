import numpy as np
import os
from .utils import check_if_uni_sorted


def calc_likelihood_bed_all(universe, chroms, model_folder, name,
                            s_index, e_index=None):
    """
    Calculate likelihood of universe for given type of model
    :param str  universe: path to universe file
    :param list chroms: list of chromosomes present in likelihood model
    :param str model_folder: path to folder with model
    :param str name: suffix of model file name, which contains information
     about model type
    :param int s_index: from which position in univers line take assess region
     start position
    :param int e_index: from which position in univers line take assess region
     end position
    :return float: likelihood of univers for given model
    """
    curent_chrom = ""
    missing_chrom = ""
    empty_start = 0
    res = 0
    e = 0
    prob_array = None
    with open(universe) as uni:
        for i in uni:
            e += 1
            i = i.split("\t")
            i[1], i[2] = int(i[1]), int(i[2])
            if len(i) > 6:
                i[6], i[7] = int(i[6]), int(i[7])
            if i[0] == missing_chrom:
                pass
            else:
                if i[0] != curent_chrom:
                    if i[0] in chroms:
                        if e != 1:
                            res += np.sum(prob_array[empty_start:, 0])
                        curent_chrom = i[0]
                        model_file = os.path.join(model_folder,
                                                  f"{curent_chrom}_{name}.npz")
                        prob_array = np.load(model_file)
                        a = prob_array.files[0]
                        prob_array = prob_array[a]
                        empty_start = 0
                    else:
                        print(f"Chromosome {i[0]} missing from model")
                        missing_chrom = i[0]
                if e_index is None:
                    end = i[s_index] + 1
                else:
                    end = i[e_index]
                r1 = np.sum(prob_array[i[s_index]:end, 1])
                r2 = np.sum(prob_array[empty_start:i[s_index], 0])
                res += r1
                res += r2
                empty_start = end
    res += np.sum(prob_array[empty_start:, 0])
    return res


def flexible_universe_likelihood(model_folder, universe,
                                 start="starts", end="ends", core="core"):
    """
    Calculate likelihood of flexible universe based on core, start,
     end coverage model
    :param str model_folder: path to folder containing model
    :param str universe: path to universe
    :param str start: model of starts file name
    :param str end: model of end file name
    :param str core: model of core file name
    :return float: likelihood
    """
    check_if_uni_sorted(universe)
    model_files = os.listdir(model_folder)
    chroms = list(set([i.split("_")[0] for i in model_files]))
    s = calc_likelihood_bed_all(universe, chroms, model_folder, start,
                                1, 6)
    e = calc_likelihood_bed_all(universe, chroms, model_folder, end,
                                7, 2)
    c = calc_likelihood_bed_all(universe, chroms, model_folder, core,
                                6, 7)
    return sum([s, e, c])


def simple_universe_likelihood(model_folder, universe,
                               start="starts", end="ends", core="core"):
    """
    Calculate likelihood of hard universe based on core, start,
    end coverage model
    :param str model_folder: path to folder containing model
    :param str universe: path to universe
    :param str start: model of starts file name
    :param str end: model of end file name
    :param str core: model of core file name
    :return float: likelihood
    """
    check_if_uni_sorted(universe)
    model_files = os.listdir(model_folder)
    chroms = list(set([i.split("_")[0] for i in model_files]))
    s = calc_likelihood_bed_all(universe, chroms, model_folder, start,
                                1)
    e = calc_likelihood_bed_all(universe, chroms, model_folder, end,
                                2)
    c = calc_likelihood_bed_all(universe, chroms, model_folder, core,
                                1, 2)
    return sum([s, e, c])


def likelihood_only_core(model_folder, universe, core="core"):
    """
    Calculate likelihood of universe based on core coverage model
    :param str model_folder: path to folder containing model
    :param str universe: path to universe
    :param str core: model file name
    :return float: likelihood
    """
    check_if_uni_sorted(universe)
    model_files = os.listdir(model_folder)
    chroms = list(set([i.split("_")[0] for i in model_files]))
    c = calc_likelihood_bed_all(universe, chroms, model_folder, core,
                                1, 2)
    return c
