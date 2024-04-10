import os

import numpy as np
import pytest
from geniml.search import vec_pairs
from geniml.search.backends import HNSWBackend
from geniml.text2bednn.text2bednn import Vec2VecFNN
from geniml.text2bednn.utils import metadata_dict_from_csv

DATA_FOLDER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data"
)

np.random.seed(100)


@pytest.fixture
def csv_path():
    """

    Returns: path of the testing metadata csv file

    """
    return os.path.join(DATA_FOLDER_PATH, "ATAC_hg38_sample_for_geniml.csv")


@pytest.fixture
def col_names():
    """
    :return: the columns that are needed metadata csv
    """
    return {
        "tissue",
        "cell_line",
        "tissue_lineage",
        "tissue_description",
        "diagnosis",
        "sample_name",
        "antibody",
    }


@pytest.fixture
def nl_payloads():
    """A list of mock payloads containing natural language metadata and matching BED files"""
    return [
        {"text": "heart muscle", "files": ["b1.bed", "b2.bed", "b3.bed"]},
        {"text": "ipf", "files": ["b1.bed", "b3.bed"]},
        {"text": "healthy control", "files": ["b2.bed"]},
    ]


@pytest.fixture
def bed_payloads():
    """A list of mock payloads containing BED file names"""
    return [
        {"name": "b1.bed"},
        {"name": "b2.bed"},
        {"name": "b3.bed"},
        {"name": "b4.bed"},
        {"name": "b5.bed"},
        {"name": "b6.bed"},
    ]


def test_metadata_dict(csv_path, col_names):
    """
    Test reading metadata from csv file
    """
    metadata_dict1 = metadata_dict_from_csv(csv_path, col_names)
    assert metadata_dict1

    metadata_dict2 = metadata_dict_from_csv(csv_path, col_names, chunk_size=1)
    assert metadata_dict2

    assert metadata_dict1 == metadata_dict2


def test_vec_pair(nl_payloads, bed_payloads, tmp_path_factory):
    """
    Test extracting vector pairs for training/validating from backends
    """
    # mock vectors
    nl_vecs = []
    bed_vecs = []
    for i in range(3):
        nl_vecs.append(np.random.random((3,)))

    for i in range(6):
        bed_vecs.append(np.random.random((2,)))

    temp_data_dir = tmp_path_factory.mktemp("data")
    bed_idx_path = temp_data_dir / "bed_idx.bin"
    nl_idx_path = temp_data_dir / "nl_idx.bin"

    bed_backend = HNSWBackend(local_index_path=str(bed_idx_path), payloads={}, dim=2)
    nl_backend = HNSWBackend(
        local_index_path=str(nl_idx_path),
        payloads={},
        dim=3,
    )

    bed_backend.load(vectors=np.array(bed_vecs), payloads=bed_payloads)
    nl_backend.load(vectors=np.array(nl_vecs), payloads=nl_payloads)

    # only target pairs
    X, Y, target = vec_pairs(nl_backend, bed_backend, "files", "name")

    assert X.shape[0] == 6
    assert Y.shape[0] == 6

    # target & non-target pairs
    X, Y, target = vec_pairs(nl_backend, bed_backend, "files", "name", True, 1.0)

    assert X.shape[0] == 12
    assert Y.shape[0] == 12

    assert (target == 1).sum() == 6
    assert (target == -1).sum() == 6


def test_torch_running(tmp_path_factory):
    """
    Test model training
    """
    training_X = np.random.random((90, 1024))
    training_Y = np.random.random((90, 100))
    training_target = np.random.choice([1, -1], size=90)
    validating_X = np.random.random((10, 1024))
    validating_Y = np.random.random((10, 100))
    validating_target = np.random.choice([1, -1], size=10)

    best_embed_folder = tmp_path_factory.mktemp("best_embed")

    v2v_torch1 = Vec2VecFNN()

    v2v_torch1.train(
        training_X,
        training_Y,
        (validating_X, validating_Y),
        save_best=False,
        folder_path=best_embed_folder,
        early_stop=True,
        patience=0.1,
        loss_func="cosine_embedding_loss",
        num_epochs=100,
        batch_size=16,
        num_units=[512, 256],
    )
    v2v_torch1.plot_training_hist(best_embed_folder)
    v2v_torch1.export(best_embed_folder, "v2v_best_epoch.pt")

    v2v_torch2 = Vec2VecFNN()
    v2v_torch2.load_from_disk(
        os.path.join(best_embed_folder, "v2v_best_epoch.pt"),
        os.path.join(best_embed_folder, "config.yaml"),
    )

    input_vecs = np.random.random((5, 1024))

    output1 = v2v_torch1.embedding_to_embedding(input_vecs)
    output2 = v2v_torch2.embedding_to_embedding(input_vecs)

    assert np.array_equal(output1, output2)

    # train the model without validate data
    v2v_torch2.train(training_X, training_Y, num_epochs=10)

    # train the model with contrastiv loss
    v2v_torch_contrast = Vec2VecFNN()

    v2v_torch_contrast.train(
        training_X,
        training_Y,
        (validating_X, validating_Y),
        save_best=False,
        folder_path=best_embed_folder,
        early_stop=True,
        patience=0.1,
        loss_func="cosine_embedding_loss",
        num_epochs=100,
        batch_size=16,
        num_units=[512, 256],
        training_target=training_target,
        validating_target=validating_target,
    )
