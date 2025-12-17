import random
from typing import Dict, List, Tuple, Union

import numpy as np


def verify_load_inputs(
    vectors: np.ndarray,
    ids: Union[List[Union[str, int]], np.ndarray],
    payloads: List[Dict[str, str]],
):
    n_ids = len(ids)
    n_vectors = vectors.shape[0]
    n_payloads = len(payloads)
    if n_ids != n_vectors or n_ids != n_payloads:
        raise ValueError(
            "The number of ids ({n_ids}), vectors ({n_vectors}), and payloads ({n_payloads}) must match"
        )


def single_query_eval(search_results: List, relevant_results: List) -> Tuple[float, float, float]:
    """Evaluate a single query's retrieval performance.

    Args:
        search_results: List of store ids ordered by similarity in search
        relevant_results: List of store ids that are relevant search results

    Returns:
        A tuple of (Average Precision, AUC-ROC, R-precision) scores.
    """
    num_relevant = len(relevant_results)
    retrieved_relevant = 0
    k = len(search_results)
    sum_precision = 0
    x = [0]  # (false_positive/(false_positive + true_negative)
    y = [0]  # recall or  true_positive / (true_positive + false_negative)
    false_positive = 0
    true_negative = k - num_relevant
    true_positive = 0
    false_negative = num_relevant

    for i in range(k):
        result = search_results[i]
        result_id = result["id"]
        if result_id in relevant_results:  # one relevant is retrieved
            true_positive += 1
            false_negative -= 1
            retrieved_relevant += 1

            sum_precision += retrieved_relevant / (i + 1)

        else:  # one irrelevant is retrieved
            false_positive += 1
            true_negative -= 1
        x.append(false_positive / (false_positive + true_negative))
        y.append(true_positive / (true_positive + false_negative))
        if i == num_relevant - 1:
            r_precision = retrieved_relevant / num_relevant
    average_precision = sum_precision / num_relevant
    # compute AUC-ROC
    auc = np.trapz(y, x)
    return average_precision, auc, r_precision


def rand_eval(n: int, query_dict: Dict) -> Tuple[float, float, float]:
    """Compute evaluation scores for random retrieval performance.

    Args:
        n: total number of results

        query_dict: a dictionary containing queries and relevant results in this format:
            {
                <query string>: [
                    <store id in backend>,
                    ...
                ],
                ...
            }

    Returns:
        A tuple of (Average Precision, AUC-ROC, R-precision) scores.
    """
    sum_ap = 0  # sum of all average precisions
    sum_auc = 0
    sum_rp = 0
    query_count = 0

    for query_str in query_dict.keys():
        relevant_results = query_dict[query_str]  # set of relevant ids
        search_results_ids = list(range(n))
        random.shuffle(search_results_ids)
        search_results = []
        for result_id in search_results_ids:
            search_results.append({"id": result_id})
        query_count += 1
        ap, auc, rp = single_query_eval(search_results, relevant_results)
        sum_ap += ap
        sum_auc += auc
        sum_rp += rp

    if query_count > 0:
        return sum_ap / query_count, sum_auc / query_count, sum_rp / query_count

    else:
        return 0.0, 0.0, 0.0
