import glob
import re
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from scipy import stats

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

# Label embedding

path_model = "../outputs/bedembed_output/starspace_model_hg19.tsv"

label_prefix = "__label__"


def label_preprocessing(path_label_embedding, label_prefix, common_labels=[]):
    labels = []
    label_vectors = []
    label_embedding = pd.read_csv(path_label_embedding, sep="\t", header=None, skiprows=1)
    vectors = label_embedding[label_embedding[0].str.contains(label_prefix)]  # .reset_index()

    vectors[0] = vectors[0].str.replace(label_prefix, "")

    if len(common_labels) > 0:
        vectors = vectors[vectors[0].isin(common_labels)]

    return vectors[list(vectors)[1:]].values, list(vectors[0])


# This function reduce the dimension using umap and plot
def UMAP_plot(
    data_X,
    y,
    title="",
    nn=100,
    metric="cosine",
    filename="",
    plottitle="",
    output_folder="",
):
    np.random.seed(3)
    dp = 400

    ump = umap.UMAP(
        a=None,
        angular_rp_forest=False,
        b=None,
        force_approximation_algorithm=False,
        learning_rate=0.9,
        local_connectivity=1.0,
        low_memory=False,
        metric=metric,
        metric_kwds=None,
        min_dist=0.1,
        n_components=2,
        n_epochs=1000,
        n_neighbors=nn,
        negative_sample_rate=3,
        output_metric="euclidean",
        output_metric_kwds=None,
        random_state=3,
        repulsion_strength=1.0,
        set_op_mix_ratio=1.0,
        spread=1.0,
        target_metric="categorical",
        target_metric_kwds=None,
        target_n_neighbors=-1,
        target_weight=0.5,
        transform_queue_size=4.0,
        transform_seed=3,
        unique=False,
        verbose=False,
    )

    ump.fit(data_X)
    ump_data = pd.DataFrame(ump.transform(data_X))

    ump_data = pd.DataFrame({"UMAP 1": ump_data[0], "UMAP 2": ump_data[1], title: y})

    ump_data = ump_data.sort_values(by=title)

    plate = sns.color_palette("husl", n_colors=len(set(y)))
    kind = "scatter"
    fig = ump_data.plot(
        kind=kind,
        x="UMAP 1",
        y="UMAP 2",
        rot=45,
        fontsize=15,
        s=500,
        title=title,
        color=plate,
        figsize=(10, 10),
        legend=True,
    )

    for i, txt in enumerate(list(ump_data[title])):
        fig.annotate(
            txt,
            (
                ump_data.iloc[i]["UMAP 1"] - 0.05,
                ump_data.iloc[i]["UMAP 2"] + 0.05,
            ),
        )
    #     plt.legend(loc='upper right', )

    return fig


from scipy.cluster import hierarchy

nn = 5
target = "target"
common_labels = []
label_vectors, labels = label_preprocessing(
    glob.glob(path_model.format(target))[0], label_prefix, common_labels
)
ump = umap.UMAP(
    a=None,
    angular_rp_forest=False,
    b=None,
    force_approximation_algorithm=False,
    learning_rate=0.9,
    local_connectivity=1.0,
    low_memory=False,
    metric="cosine",
    metric_kwds=None,
    min_dist=0.1,
    n_components=2,
    n_epochs=1000,
    n_neighbors=nn,
    negative_sample_rate=3,
    output_metric="euclidean",
    output_metric_kwds=None,
    random_state=3,
    repulsion_strength=1.0,
    set_op_mix_ratio=1.0,
    spread=1.0,
    target_metric="categorical",
    target_metric_kwds=None,
    target_n_neighbors=-1,
    target_weight=0.5,
    transform_queue_size=4.0,
    transform_seed=3,
    unique=False,
    verbose=False,
)

ump.fit(label_vectors)
ump_data = pd.DataFrame(ump.transform(label_vectors))


Z = hierarchy.linkage(ump_data, "weighted")
hierarchy.dendrogram(Z, labels=labels, leaf_rotation=90)
plt.show()


target = "target"
common_labels = []

fig = UMAP_plot(
    label_vectors,
    labels,
    title="label_embedding",
    metric="cosine",
    nn=nn,
    filename="",
    plottitle="Label Embedding",
    output_folder="./",
)


# Scenario 1

path_meta = "../"
meta_data = pd.read_csv(path_meta + "tests/test_file_meta.csv")
cell_types = list(set(meta_data["cell_type"]))
targets = list(set(meta_data["target"]))
targets = [str(t).lower() for t in targets]


def retrieve_meta_test():
    meta_test = (pd.read_csv(path_meta + "tests/test_file_meta.csv"))[
        ["file_name", "cell_type", "target"]
    ]
    meta_test["original_label"] = meta_test["target"] + " " + meta_test["cell_type"]
    return meta_test


meta_test = pd.concat([retrieve_meta_test()])
meta_test.file_name = "/project/shefflab/data/encode/" + meta_test.file_name

path_simfile = "../outputs/bedembed_output/similarity_score_hg19.csv"
search = "target"


def Scenario1(path_simfile):
    distance = pd.read_csv(file)
    distance.file_label = distance.file_label.str.lower()
    distance.search_term = distance.search_term.str.lower()
    distance = distance.drop_duplicates()
    search_table = pd.pivot_table(
        distance, values="score", index=["filename"], columns=["search_term"]
    ).reset_index()
    search_table = pd.merge(
        distance[["filename", "file_label"]].drop_duplicates(),
        search_table,
        on="filename",
    ).drop_duplicates()
    search_table = search_table.merge(meta_test, left_on="filename", right_on="file_name")

    if search == "cell":
        ind = 0
        training_labels = cell_types
    else:
        ind = 1
        training_labels = targets

    training_labels = sorted(training_labels)

    ### ???
    training_labels = list(search_table)[2:-4]

    len_targets = len(search_table.file_label[0].split(" _")) - 1

    search_table.file_label = search_table.file_label.str.split(" _", expand=True)[
        np.min([ind, len_targets])
    ]

    search_table = search_table[["filename", "file_label", "original_label"] + (training_labels)]
    search_table["predicted_label"] = search_table[list(search_table)[3:]].idxmin(axis=1)

    for searchterm in training_labels:
        nof = len(search_table[search_table.file_label.str.contains(searchterm)])
        df = search_table[["filename", "file_label", "original_label", searchterm]].sort_values(
            by=[searchterm]
        )[0:10]
        df = df.sort_values(by=[searchterm], ascending=False)

        df["color"] = "gray"
        df.loc[df.file_label.str.contains(searchterm), "color"] = "green"
        if len(df[df.color == "green"]) == nof:
            df.loc[(df.color != "green"), "color"] = "gray"

        df[searchterm] = 1 - df[searchterm]

        plt = df.plot.barh(
            x="original_label",
            y=searchterm,
            figsize=(10, 7),
            fontsize=16,
            color=list(df["color"]),
        )
        plt.set_xlabel("Similarity", fontsize=15)
        plt.set_ylabel("original_label", fontsize=15)

        plt.axis(xmin=0.5, xmax=1.01)

        plt.figure.savefig(
            "../outputs/bedembed_output/figures/S1/{}_nof{}.svg".format(searchterm, nof),
            format="svg",
            bbox_inches="tight",
        )


Scenario1(path_simfile)
# End of Scenario 1


# Scenario 2


search = "target"


def Scenario2(path_simfile):
    distance = pd.read_csv(file)
    distance.file_label = distance.file_label.str.lower()
    distance.search_term = distance.search_term.str.lower()
    distance = distance.drop_duplicates()
    search_table = pd.pivot_table(
        distance, values="score", index=["filename"], columns=["search_term"]
    ).reset_index()
    search_table = search_table.merge(meta_test, left_on="filename", right_on="file_name")
    search_table = pd.merge(
        distance[["filename", "file_label"]].drop_duplicates(),
        search_table,
        on="filename",
    ).drop_duplicates()

    if "cell" in search:
        ind = 0
        training_labels = cell_types
    else:
        ind = 1
        training_labels = targets

    training_labels = sorted(training_labels)

    len_targets = len(search_table.file_label[0].split(" _")) - 1

    search_table.file_label = search_table.file_label.str.split(" _", expand=True)[
        np.min([ind, len_targets])
    ]

    search_table = search_table[["filename", "file_label", "original_label"] + (training_labels)]
    search_table["predicted_label"] = search_table[list(search_table)[3:]].idxmin(axis=1)

    i = 0
    b = search_table
    all_weights = []
    for fil in b.filename:
        c = b[b.filename == fil]["file_label"]
        i = c.index[0]

        a = b[b.filename == fil][list(b[b.filename == fil])[3:-1]]
        weights = []

        ol = b[b.filename == fil]["file_label"][i]
        for lb in list(a):
            if (a[lb][i]) == 0:
                weights.append((fil, ol, lb, (1 - a[lb][i])))
            else:
                weights.append((fil, ol, lb, (1 - a[lb][i])))

        all_weights.extend(weights)

        i += 1

    X = pd.DataFrame(all_weights).rename(
        columns={
            0: "Filename",
            1: "Filelabel",
            2: "AllLabels",
            3: "Distance_score",
        }
    )

    for file in list(set(X.Filename)):
        df = X[X.Filename == file].sort_values(by=["Distance_score"], ascending=False)[0:10]
        df = df.sort_values(by=["Distance_score"], ascending=True)
        df["color"] = "green"
        plt = df.plot.barh(
            x="AllLabels",
            y="Distance_score",
            figsize=(8, 5),
            fontsize=16,
            color=list(df["color"]),
        )
        plt.set_xticks(np.arange(0.5, 1.1, 0.1))
        plt.set_xlabel("Similarity", fontsize=15)
        plt.set_ylabel("original_label", fontsize=15)

        plt.figure.savefig(
            "../outputs/bedembed_output/figures/S2/" + file.split("/")[-1] + ".svg",
            format="svg",
            bbox_inches="tight",
            dpi=300,
        )


Scenario2(path_simfile)
# End of Scenario 2


# Scenario 3

meta_test = (pd.read_csv(path_meta + "tests/test_file_meta.csv"))[
    ["file_name", "cell_type", "target"]
]
meta_test["original_label_test"] = meta_test["target"] + " " + meta_test["cell_type"]
meta_train = (pd.read_csv(path_meta + "tests/test_file_meta.csv"))[
    ["file_name", "cell_type", "target"]
]
meta_train["original_label_train"] = meta_train["target"] + " " + meta_train["cell_type"]
meta_test.file_name = "/project/shefflab/data/encode/" + meta_test.file_name
meta_train.file_name = "/project/shefflab/data/encode/" + meta_train.file_name

file_name = "../outputs/bedembed_output/query_db_similarity_score_hg19.csv"

sim = pd.read_csv(file_name)

sim[["train_name", "train_label_cell", "train_label_target"]] = sim.db_file.str.split(
    ",", expand=True
)[[0, 1, 2]]
sim[["test_name", "test_label_cell", "test_label_target"]] = sim.test_file.str.split(
    ",", expand=True
)[[0, 1, 2]]

sim = sim.merge(meta_test, left_on="test_name", right_on="file_name").merge(
    meta_train,
    left_on="train_name",
    right_on="file_name",
    suffixes=("_test", "_train"),
)  #

sim.score = 1 - sim.score

for test in list(set(sim.test_name)):
    df = sim[sim.test_name == test].sort_values(by="score", ascending=False)[
        [
            "test_name",
            "train_name",
            "original_label_test",
            "original_label_train",
            "score",
        ]
    ]
    nof = len(df[df.original_label_test == df.original_label_train])
    df = df[0:10]

    df = df.sort_values(by=["score"])
    df["color"] = "gray"
    df.loc[df.original_label_test == df.original_label_train, "color"] = "green"

    if len(df[df.color == "green"]) == nof:
        df.loc[(df.color != "green"), "color"] = "gray"

    plt = df.plot.barh(
        x="original_label_train",
        y="score",
        figsize=(10, 7),
        fontsize=16,
        color=list(df["color"]),
    )
    plt.axis(xmin=0.7, xmax=1.01)

    plt.figure.savefig(
        "../outputs/bedembed_output/figures/S3/{}_nof{}.svg".format(test.split("/")[-1], nof),
        format="svg",
        bbox_inches="tight",
    )

# End of Scenario 3
