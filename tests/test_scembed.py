import logging
import os
import sys

import pytest
import scanpy as sc
from genimtools.utils import write_tokens_to_gtok
from tqdm import tqdm

from geniml.region2vec.utils import Region2VecDataset
from geniml.scembed.main import ScEmbed
from geniml.tokenization.main import AnnDataTokenizer

# add parent directory to path
sys.path.append("../")


# set to DEBUG to see more info
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def universe_file():
    return "tests/data/universe.bed"


@pytest.fixture
def hf_model():
    return "databio/r2v-ChIP-atlas-hg38-v2"


@pytest.fixture
def pbmc_data():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad")


@pytest.fixture
def pbmc_data_backed():
    return sc.read_h5ad("tests/data/pbmc_hg38.h5ad", backed="r")


def test_model_creation():
    model = ScEmbed()
    assert model


def test_model_training(universe_file: str, pbmc_data: sc.AnnData):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = ScEmbed(tokenizer=AnnDataTokenizer(universe_file))  # set to 1 for testing

    dataset = Region2VecDataset("tests/data/gtok_sample/", convert_to_str=True)
    model.train(dataset, epochs=3, min_count=1)

    # keep only columns with values > 0
    pbmc_data = pbmc_data[:, pbmc_data.X.sum(axis=0) > 0]

    assert model.trained


def test_model_train_and_export(universe_file: str):
    # remove gensim logging
    logging.getLogger("gensim").setLevel(logging.ERROR)
    model = ScEmbed(tokenizer=AnnDataTokenizer(universe_file))  # set to 1 for testing

    dataset = Region2VecDataset("tests/data/gtok_sample/", convert_to_str=True)
    model.train(dataset, epochs=3, min_count=1)

    assert model.trained

    # save
    try:
        model.export("tests/data/model-tests")
        model = ScEmbed.from_pretrained("tests/data/model-tests")

        # ensure model is still trained and has region2vec
        assert model.trained

    finally:
        os.remove("tests/data/model-tests/checkpoint.pt")
        os.remove("tests/data/model-tests/universe.bed")
        os.remove("tests/data/model-tests/config.yaml")


@pytest.mark.skip(reason="Need to get a pretrained model first")
def test_pretrained_scembed_model(hf_model: str, pbmc_data: sc.AnnData):
    model = ScEmbed(hf_model)
    embeddings = model.encode(pbmc_data)
    assert embeddings.shape[0] == pbmc_data.shape[0]


@pytest.mark.skip(reason="This is for my own testing")
def test_end_to_end_training():
    import matplotlib.pyplot as plt
    from umap import UMAP

    data_path = os.path.expandvars("$HOME/Desktop/buenrostro2018.h5ad")
    tokens_path = os.path.expandvars("$HOME/Desktop/tokens")
    universe_path = os.path.expandvars("$HOME/Desktop/universe.bed")
    export_path = os.path.expandvars("$HOME/Desktop/scembed-model")

    # try to remove tokens folder if it exists
    if os.path.exists(tokens_path):
        os.system(f"rm -rf {tokens_path}")

    adata = sc.read_h5ad(data_path)

    tokenizer = AnnDataTokenizer(universe_path)
    tokens = tokenizer.tokenize(adata)

    for i, t in tqdm(enumerate(tokens)):
        file = os.path.join(tokens_path, f"cell{i}.gtok")
        write_tokens_to_gtok(file, t.ids)

    dataset = Region2VecDataset(tokens_path, convert_to_str=True, shuffle=True)

    model = ScEmbed(tokenizer=tokenizer)
    model.train(dataset, epochs=3, num_cpus=5)

    model.export(export_path)

    model2 = ScEmbed.from_pretrained(export_path)
    embeddings = model2.encode(adata)

    reducer = UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=0.1)
    fig.savefig("umap_test.png")
