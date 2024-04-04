import lightning as L
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from geniml.region2vec.main import Region2Vec, Region2VecExModel
from geniml.tokenization.main import ITTokenizer
from geniml.training import CellTypeFineTuneAdapter
from geniml.training.utils import (
    FineTuningDataset,
    collate_finetuning_batch,
    generate_fine_tuning_dataset,
)


def test_generate_finetuning_dataset():
    t = ITTokenizer("tests/data/universe.bed")
    adata = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")

    pos, neg, pos_labels, neg_labels = generate_fine_tuning_dataset(
        adata, t, negative_ratio=1.0, sample_size=2
    )

    # total positive pairs should be equal to total negative pairs
    # total positive pairs will be equal to sum([n*(n - 1) for n in adata.obs.groupby("cell_type").size()])
    # total negative pairs only equals number of positive pairs when negative_ratio=1.0

    assert len(pos) == len(neg)
    assert len(pos) == len(pos_labels)
    assert len(neg) == len(neg_labels)
    # not sure why the below doesnt work right now
    # assert len(pos) == sum([(n * (n - 1)) for n in adata.obs.groupby("cell_type").size()])


def test_init_celltype_adapter():
    model = Region2VecExModel(
        tokenizer="tests/data/universe.bed",
    )
    adapter = CellTypeFineTuneAdapter(model)
    assert adapter is not None
    assert isinstance(adapter.nn_model, Region2Vec)
    assert adapter.nn_model.projection.num_embeddings == len(model.tokenizer)


def test_train_with_adapter():
    # make models
    model = Region2VecExModel(
        tokenizer="tests/data/universe.bed",
    )
    adapter = CellTypeFineTuneAdapter(model)

    # load data
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    pos_pairs, neg_pairs, pos_labels, neg_labels = generate_fine_tuning_dataset(
        data, model.tokenizer, seed=42, negative_ratio=1.0, sample_size=1_000
    )

    # combine the positive and negative pairs
    pairs = pos_pairs + neg_pairs
    labels = pos_labels + neg_labels

    # get the pad token id
    pad_token_id = model.tokenizer.padding_token_id()

    train_pairs, test_pairs, Y_train, Y_test = train_test_split(
        pairs,
        labels,
        train_size=0.8,
        random_state=42,
    )

    batch_size = 32

    # create the datasets
    train_dataloader = DataLoader(
        FineTuningDataset(train_pairs, Y_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_finetuning_batch(x, pad_token_id),
        # num_workers=multiprocessing.cpu_count() - 2,
    )
    test_dataloader = DataLoader(
        FineTuningDataset(test_pairs, Y_test),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_finetuning_batch(x, pad_token_id),
        # num_workers=multiprocessing.cpu_count() - 2,
    )

    trainer = L.Trainer(profiler="simple", min_epochs=3)
    trainer.fit(adapter, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)


def test_train_export():
    # make models
    model = Region2VecExModel(
        tokenizer="tests/data/universe.bed",
    )
    adapter = CellTypeFineTuneAdapter(model)

    # load data
    data = sc.read_h5ad("tests/data/pbmc_hg38.h5ad")
    pos_pairs, neg_pairs, pos_labels, neg_labels = generate_fine_tuning_dataset(
        data, model.tokenizer, seed=42, negative_ratio=1.0, sample_size=1_000
    )

    # combine the positive and negative pairs
    pairs = pos_pairs + neg_pairs
    labels = pos_labels + neg_labels

    # get the pad token id
    pad_token_id = model.tokenizer.padding_token_id()

    train_pairs, test_pairs, Y_train, Y_test = train_test_split(
        pairs,
        labels,
        train_size=0.8,
        random_state=42,
    )

    batch_size = 32

    # create the datasets
    train_dataloader = DataLoader(
        FineTuningDataset(train_pairs, Y_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_finetuning_batch(x, pad_token_id),
        # num_workers=multiprocessing.cpu_count() - 2,
    )
    test_dataloader = DataLoader(
        FineTuningDataset(test_pairs, Y_test),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_finetuning_batch(x, pad_token_id),
        # num_workers=multiprocessing.cpu_count() - 2,
    )

    trainer = L.Trainer(min_epochs=3)
    trainer.fit(adapter, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)

    # get the tensor dfor token 42
    t_before = model._model.projection(torch.tensor([42]))

    # export
    model.export("tests/data/model-tests")

    # load the model
    model = Region2VecExModel.from_pretrained("tests/data/model-tests")

    # get the tensor for token 42
    t_after = model._model.projection(torch.tensor([42]))

    # make sure the tensors are close
    assert torch.allclose(t_before, t_after)
