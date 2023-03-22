import numpy as np
import os
from .utils import check_if_uni_sorted
from ..likelihood.build_model import ChromosomeModel



def calc_likelihood_hard(universe, chroms, model_folder, name,
                         s_index, e_index=None):
    """
    Calculate likelihood of universe for given type of model
    To be used with binomial model
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
            if i[0] == missing_chrom:
                pass
            else:
                if i[0] != curent_chrom:
                    if i[0] in chroms:
                        if e != 1:
                            res += np.sum(prob_array[empty_start:, 0])
                        curent_chrom = i[0]
                        chrom_model = ChromosomeModel(model_folder, curent_chrom)
                        chrom_model.read_track(name)
                        prob_array = chrom_model.models[name]
                        empty_start = 0
                    else:
                        print(f"Chromosome {i[0]} missing from model")
                        missing_chrom = i[0]
                start = i[s_index]
                if e_index is None:
                    end = i[s_index] + 1
                else:
                    end = i[e_index]
                r1 = np.sum(prob_array[start:end, 1])
                r2 = np.sum(prob_array[empty_start:start, 0])
                res += r1
                res += r2
                empty_start = end
    res += np.sum(prob_array[empty_start:, 0])
    return res


def hard_universe_likelihood(model_folder, universe):
    """
    Calculate likelihood of hard universe based on core, start,
    end coverage model
    :param str model_folder: path to folder containing model
    :param str universe: path to universe
    :return float: likelihood
    """
    check_if_uni_sorted(universe)
    model_files = os.listdir(model_folder)
    chroms = list(set([i.split("_")[0] for i in model_files]))
    s = calc_likelihood_hard(universe, chroms, model_folder, "start",
                             1)
    e = calc_likelihood_hard(universe, chroms, model_folder, "end",
                             2)
    c = calc_likelihood_hard(universe, chroms, model_folder, "core",
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
    c = calc_likelihood_hard(universe, chroms, model_folder, core,
                             1, 2)
    return c


def background_likelihood(start, end, model_start, model_cove, model_end):
    res = np.sum(model_start[start: end, 0])
    res += np.sum(model_cove[start: end, 0])
    res += np.sum(model_end[start: end, 0])
    return res


def weigh_livelihood(start, end, model_process, model_cove, model_out,
                     reverse):
    e_w = 1 / (end - start)  # weights for processed model
    c_w = np.linspace(start=e_w, stop=1, num=(end - start))  # weights for core in processed region
    if reverse:
        c_w = c_w[::-1]
    res = e_w * np.sum(model_process[start: end, 1])
    res += np.sum(c_w * model_cove[start: end, 1])
    res += (1 - e_w) * np.sum(model_process[start: end, 0])
    res += np.sum((1 - c_w) * model_cove[start: end, 0])
    res += np.sum(model_out[start: end, 0])
    return res


def flexible_peak_likelihood(startS, startE, endS, endE,
                             model_start, model_cove, model_end):
    # core part of the peak
    res = np.sum(model_cove[startE: endS, 1])
    res += np.sum(model_start[startE: endS, 0])
    res += np.sum(model_end[startE: endS, 0])
    # start part of the peak
    res += weigh_livelihood(startS, startE, model_start, model_cove,
                            model_end, False)
    # end part of the peak
    res += weigh_livelihood(endS, endE, model_end, model_cove,
                            model_start, True)
    return res


def likelihood_flexible_universe(model_folder, universe,
                                 save_peak_input=False):
    curent_chrom = ""
    missing_chrom = ""
    empty_start = 0
    res = 0
    check_if_uni_sorted(universe)
    model_files = os.listdir(model_folder)
    chroms = list(set([i.split("_")[0] for i in model_files]))
    if save_peak_input:
        output = []
    e = 0  # number of processed chromosomes
    with open(universe) as uni:
        for line in uni:
            i = line.split("\t")
            peak_start_s, peak_end_e = int(i[1]), int(i[2])
            peak_start_e, peak_end_s = int(i[6]), int(i[7])
            if i[0] == missing_chrom:
                pass
            else:
                if i[0] != curent_chrom:
                    if i[0] in chroms:
                        if e != 0:
                            # if we read any chromosomes add to result background
                            # likelihood of part of the genome after the last region
                            res += background_likelihood(empty_start, len(model_start),
                                                         model_start, model_core, model_end)
                        curent_chrom = i[0]
                        e += 1
                        chrom_model = ChromosomeModel(model_folder, curent_chrom)
                        chrom_model.read()
                        model_start = chrom_model.models["start"]
                        model_core = chrom_model.models["core"]
                        model_end = chrom_model.models["end"]

                    else:
                        print(f"Chromosome {i[0]} missing from model")
                        missing_chrom = i[0]
            res += background_likelihood(empty_start, peak_start_s,
                                         model_start, model_core, model_end)
            peak_likelihood = flexible_peak_likelihood(peak_start_s, peak_start_e, peak_end_s, peak_end_e,
                                                       model_start, model_core, model_end)
            res += peak_likelihood
            if save_peak_input:
                backgroung = background_likelihood(peak_start_s, peak_end_e,
                                                   model_start, model_core, model_end)
                contribution = peak_likelihood - backgroung
                output.append("{}\t{}\n".format(line.strip("\n"), contribution))
            empty_start = peak_end_e

        res += background_likelihood(empty_start, len(model_start),
                                     model_start, model_core, model_end)
        if save_peak_input:
            print("saving")
            with open(universe + "_peak_likelihood", "w") as f:
                f.writelines(output)
    return res
