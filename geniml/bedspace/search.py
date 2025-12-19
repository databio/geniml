import logging

import pandas as pd

from .const import DEFAULT_NUM_SEARCH_RESULTS, PKG_NAME

_LOGGER = logging.getLogger(PKG_NAME)


def run_scenario1(
    query: str,
    distances: str,
    output: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """Run the search command for scenario 1: Give me a label, I'll return region sets.

    Args:
        query (str): The query string (a label).
        distances (str): The path to the distances file.
        num_results (int): The number of results to return.
        output (str): The path to save the barplots.
    """
    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE
    searchterm = query
    distance = pd.read_csv(distances)
    distance.file_label = distance.file_label.str.lower()
    distance.search_term = distance.search_term.str.lower()
    distance = distance.drop_duplicates()
    search_table = pd.pivot_table(
        distance, values="score", index=["filename"], columns=["search_term"]
    ).reset_index()
    df = search_table[["filename", searchterm]].sort_values(by=[searchterm], ascending=False)[
        0:num_results
    ]
    df = df.sort_values(by=[searchterm], ascending=True)
    df["color"] = "green"
    plt = df.plot.barh(
        x="filename",
        y=searchterm,
        figsize=(6, 4),
        fontsize=10,
        color=list(df["color"]),
    )
    plt.set_xlabel("Similarity", fontsize=10)
    plt.set_ylabel("File_name", fontsize=10)
    plt.axis(xmin=0.0, xmax=1.01)

    plt.figure.savefig(
        "{}/{}_nof{}.svg".format(output, query, num_results),
        format="svg",
        bbox_inches="tight",
    )


def run_scenario2(
    query: str,
    distances: str,
    output: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """Run the search command for scenario 2: Give me a region set, I'll return labels.

    Args:
        query (str): The query string (a path to a file).
        distances (str): The path to the distances file.
        num_results (int): The number of results to return.
        output (str): The path to save the barplots.
    """

    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE
    file = query
    distance = pd.read_csv(distances)
    distance.file_label = distance.file_label.str.lower()
    distance.search_term = distance.search_term.str.lower()
    distance = distance.drop_duplicates()

    df = distance[distance.filename == query].sort_values(by=["score"], ascending=True)

    df["color"] = "green"
    plt = df.plot.barh(
        x="search_term",
        y="score",
        figsize=(6, 4),
        fontsize=10,
        color=list(df["color"]),
    )
    plt.set_xlabel("Similarity", fontsize=10)
    plt.set_ylabel("Ranked labels", fontsize=10)
    plt.axis(xmin=0.0, xmax=1.01)

    plt.figure.savefig(
        "{}/{}_nof{}.svg".format(output, query.split("/")[-1], num_results),
        format="svg",
        bbox_inches="tight",
    )


def run_scenario3(
    query: str,
    distances: str,
    output: str,
    num_results: int = DEFAULT_NUM_SEARCH_RESULTS,
):
    """Run the search command for scenario 3: Give me a region set, I'll return region sets.

    Args:
        query (str): The query string (a path to a file).
        distances (str): The path to the distances file.
        num_results (int): The number of results to return.
        output (str): The path to save the barplots.
    """

    _LOGGER.info("Running search...")

    # PLACE SEARCH CODE HERE

    file = query.lower()
    distance = pd.read_csv(distances)
    distance.test_file = distance.test_file.str.lower()
    distance.db_file = distance.db_file.str.lower()
    distance = distance.drop_duplicates()
    df = distance[distance.test_file.str.contains(file)].sort_values(
        by=["score"], ascending=False
    )[0:num_results]
    df = df.sort_values(by=["score"], ascending=True)

    df["color"] = "green"
    plt = df.plot.barh(
        x="db_file",
        y="score",
        figsize=(6, 4),
        fontsize=10,
        color=list(df["color"]),
    )
    plt.set_xlabel("Similarity", fontsize=10)
    plt.set_ylabel("Files in db", fontsize=10)
    plt.axis(xmin=0.0, xmax=1.01)

    plt.figure.savefig(
        "{}/{}_nof{}.svg".format(output, query.split("/")[-1], num_results),
        format="svg",
        bbox_inches="tight",
    )
