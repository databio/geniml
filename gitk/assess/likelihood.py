import numpy as np
import os
from .utils import check_if_uni_sorted, check_if_uni_flexible
from ..utils import read_chromosome_from_bw
from ..likelihood.build_model import ModelLH


class LhModel:
    def __init__(self, model, cove):
        """
        Object with combined information about lh model and coverage
        :param ndarray model: lh model array
        :param ndarray cove: coverage array
        """
        self.model = model
        self.cove = cove

    def __getitem__(self, sliced):
        return self.model[self.cove[sliced[0]], sliced[1]]


def calc_likelihood_hard(
    universe,
    chroms,
    model_lh,
    coverage_folder,
    coverage_prefix,
    name,
    s_index,
    e_index=None,
):
    """
    Calculate likelihood of universe for given type of model
    To be used with binomial model
    :param str  universe: path to universe file
    :param list chroms: list of chromosomes present in model
    :param ModelLH model_lh: likelihood model
    :param coverage_prefix: prefix used in uniwig for creating coverage
    :param coverage_folder: path to a folder with genome coverage by tracks
    :param str name: suffix of model file name, which contains information
     about model type
    :param int s_index: from which position in univers line take assess region
     start position
    :param int e_index: from which position in univers line take assess region
     end position
    :return float: likelihood of univers for given model
    """
    current_chrom = ""
    missing_chrom = ""
    empty_start = 0
    res = 0
    e = 0
    prob_array, cove_array = None, None
    with open(universe) as uni:
        for i in uni:
            e += 1
            i = i.split("\t")
            i[1], i[2] = int(i[1]), int(i[2])
            if i[0] == missing_chrom:
                pass
            else:
                if i[0] != current_chrom:
                    if i[0] in chroms:
                        model_lh.clear_chrom(current_chrom)
                        if e != 1:
                            res += np.sum(prob_array[empty_start:, 0])

                        current_chrom = i[0]
                        model_lh.read_chrom_track(current_chrom, name)
                        prob_model = model_lh[current_chrom]
                        cove_array = read_chromosome_from_bw(
                            os.path.join(
                                coverage_folder, f"{coverage_prefix}_{name}.bw"
                            ),
                            current_chrom,
                        )
                        prob_array = LhModel(prob_model[name], cove_array)
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


def hard_universe_likelihood(model, universe, coverage_folder, coverage_prefix):
    """
    Calculate likelihood of hard universe based on core, start,
    end coverage model
    :param str model: path to file containing model
    :param str universe: path to universe
    :param coverage_prefix: prefix used in uniwig for creating coverage
    :param coverage_folder: path to a folder with genome coverage by tracks
    :return float: likelihood
    """
    check_if_uni_sorted(universe)
    model_lh = ModelLH(model)
    chroms = model_lh.chromosomes_list
    s = calc_likelihood_hard(
        universe, chroms, model_lh, coverage_folder, coverage_prefix, "start", 1
    )
    e = calc_likelihood_hard(
        universe, chroms, model_lh, coverage_folder, coverage_prefix, "end", 2
    )
    c = calc_likelihood_hard(
        universe, chroms, model_lh, coverage_folder, coverage_prefix, "core", 1, 2
    )
    return sum([s, e, c])


def likelihood_only_core(model_file, universe, coverage_folder, coverage_prefix):
    """
    Calculate likelihood of universe based only on core coverage model
    :param str model_file: path to name containing model
    :param str universe: path to universe
    :param coverage_prefix: prefix used in uniwig for creating coverage
    :param coverage_folder: path to a folder with genome coverage by tracks
    :return float: likelihood
    """
    check_if_uni_sorted(universe)
    model_lh = ModelLH(model_file)
    chroms = model_lh.chromosomes_list
    c = calc_likelihood_hard(
        universe, chroms, model_lh, coverage_folder, coverage_prefix, "core", 1, 2
    )
    return c


def background_likelihood(start, end, model_start, model_cove, model_end):
    """
    Calculate likelihood of background for given region
    """
    res = np.sum(model_start[start:end, 0])
    res += np.sum(model_cove[start:end, 0])
    res += np.sum(model_end[start:end, 0])
    return res


def weigh_livelihood(start, end, model_process, model_cove, model_out, reverse):
    """
    Calculate weighted likelihood of flexible part of the region
    :param int start: start of the region
    :param int end: end of the region
    :param array model_process: model for analysed type of flexible region
    :param array model_cove: model for coverage
    :param array model_out: model for flexible region that is not being analysed
    :param bool reverse: if model_process corespondents to end we have to reverse the weighs
    :return float: likelihood of flexible part of the region
    """
    e_w = 1 / (end - start)  # weights for processed model
    c_w = np.linspace(
        start=e_w, stop=1, num=(end - start)
    )  # weights for core in processed region
    if reverse:
        c_w = c_w[::-1]
    res = e_w * np.sum(model_process[start:end, 1])
    res += np.sum(c_w * model_cove[start:end, 1])
    res += (1 - e_w) * np.sum(model_process[start:end, 0])
    res += np.sum((1 - c_w) * model_cove[start:end, 0])
    res += np.sum(model_out[start:end, 0])
    return res


def flexible_peak_likelihood(
    start_s, start_e, end_s, end_e, model_start, model_cove, model_end
):
    # core part of the peak
    res = np.sum(model_cove[start_e:end_s, 1])
    res += np.sum(model_start[start_e:end_s, 0])
    res += np.sum(model_end[start_e:end_s, 0])
    # start part of the peak
    res += weigh_livelihood(start_s, start_e, model_start, model_cove, model_end, False)
    # end part of the peak
    res += weigh_livelihood(end_s, end_e, model_end, model_cove, model_start, True)
    return res


def read_coverage(cove_folder, cove_prefix, current_chrom):
    cove_start = read_chromosome_from_bw(
        os.path.join(cove_folder, f"{cove_prefix}_start.bw"),
        current_chrom,
    )
    cove_core = read_chromosome_from_bw(
        os.path.join(cove_folder, f"{cove_prefix}_core.bw"),
        current_chrom,
    )
    cove_end = read_chromosome_from_bw(
        os.path.join(cove_folder, f"{cove_prefix}_end.bw"),
        current_chrom,
    )
    cove = {"start": cove_start, "core": cove_core, "end": cove_end}
    return cove


def likelihood_flexible_universe(
    model_file, universe, cove_folder, cove_prefix, save_peak_input=False
):
    """
    Likelihood of given universe under the model
    :param str model_file: path to file with lh model
    :param str universe: path to universe
    :param cove_folder: path to a folder with genome coverage by tracks
    :param cove_prefix: prefix used in uniwig for creating coverage
    :param bool save_peak_input: whether to save universe with each peak lh
    :return float: lh of the flexible universe
    """
    current_chrom = ""
    missing_chrom = ""
    empty_start = 0
    res = 0
    check_if_uni_sorted(universe)
    check_if_uni_flexible(universe)
    model_lh = ModelLH(model_file)
    chroms = model_lh.chromosomes_list
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
                if i[0] != current_chrom:
                    if i[0] in chroms:
                        model_lh.clear_chrom(current_chrom)
                        if e != 0:
                            # if we read any chromosomes add to result background
                            # likelihood of part of the genome after the last region
                            res += background_likelihood(
                                empty_start,
                                chr_size,
                                prob_start,
                                prob_core,
                                prob_end,
                            )
                        current_chrom = i[0]
                        e += 1
                        model_lh.read_chrom(current_chrom)
                        models_current = model_lh[current_chrom]
                        cove_current = read_coverage(
                            cove_folder, cove_prefix, current_chrom
                        )
                        chr_size = len(cove_current["start"])
                        prob_start = LhModel(
                            models_current["start"], cove_current["start"]
                        )
                        prob_core = LhModel(
                            models_current["core"], cove_current["core"]
                        )
                        prob_end = LhModel(models_current["end"], cove_current["end"])

                    else:
                        print(f"Chromosome {i[0]} missing from model")
                        missing_chrom = i[0]
            res += background_likelihood(
                empty_start,
                peak_start_s,
                prob_start,
                prob_core,
                prob_end,
            )
            peak_likelihood = flexible_peak_likelihood(
                peak_start_s,
                peak_start_e,
                peak_end_s,
                peak_end_e,
                prob_start,
                prob_core,
                prob_end,
            )
            res += peak_likelihood
            if save_peak_input:
                background = background_likelihood(
                    peak_start_s, peak_end_e, prob_start, prob_core, prob_end
                )
                contribution = peak_likelihood - background
                output.append("{}\t{}\n".format(line.strip("\n"), contribution))
            empty_start = peak_end_e

        res += background_likelihood(
            empty_start, chr_size, prob_start, prob_core, prob_end
        )
        if save_peak_input:
            print("saving")
            with open(universe + "_peak_likelihood", "w") as f:
                f.writelines(output)
    return res
