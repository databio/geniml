from .distance import run_distance
from .intersection import run_intersection
from .likelihood import hard_universe_likelihood, likelihood_flexible_universe
import pandas as pd
import os
import numpy as np
import warnings


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
    if not all(
        [
            overlap,
            distance_f_t_u,
            distance_f_t_u_flex,
            distance_u_t_f,
            distance_u_t_f_flex,
        ]
    ):
        raise AttributeError("Choose at least one assessment method")
    if not all(
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
        r_distance_utf_flex.columns = ["file", "median_dist_universe_to_file_flex"]
        asses_results.append(r_distance_utf_flex)
    df = asses_results[0]
    for i in asses_results[1:]:
        df = pd.merge(df, i, on="file")
    df.to_csv(os.path.join(folder_out, pref + "_data.csv"), index=False)


def get_closeness_score(folder, file_list, universe, no_workers, flexible=False):
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
    me = (10 * file_to_uni[1] + uni_to_file[1]) / 11
    cs = 1001 / (me + 1000) / 1.001
    return np.mean(cs)


def get_closeness_score_from_assessment_file(file, cs_each_file=False):
    df = pd.read_csv(file, index_col=(0))
    df = df[["median_dist_file_to_universe", "median_dist_universe_to_file"]]
    df["CS"] = (
        df["median_dist_universe_to_file"] + 10 * df["median_dist_file_to_universe"]
    ) / 11
    df["CS"] = (1001 / (df["CS"] + 1000)) / 1.001
    if cs_each_file:
        return df
    else:
        return df["CS"].mean()


def get_f_10_score(
    folder,
    file_list,
    universe,
    no_workers,
):
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
    cove_prefix,
    flexible=False,
    save_peak_input=False,
):
    if flexible:
        lh = likelihood_flexible_universe(
            model_file, universe, cove_folder, cove_prefix, save_peak_input
        )
    else:
        if save_peak_input:
            warnings.warn("Unused argument: save_peak_input")
        lh = hard_universe_likelihood(model_file, universe, cove_folder, cove_prefix)

    return lh
