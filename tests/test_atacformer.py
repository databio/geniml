import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

from geniml.atacformer.main import Atacformer, AtacformerExModel
from geniml.atacformer.utils import AtacformerMLMDataset
from geniml.tokenization.main import AnnDataTokenizer
from geniml.training.adapters import MLMAdapter


@pytest.fixture
def universe_file():
    return "tests/data/universe_mlm.bed"


@pytest.fixture
def data():
    return "tests/data/gtok_sample/"


@pytest.mark.skip("Too new to test")
def test_atacformer_dataset():
    path_to_data = "tests/data/gtok_sample/"
    dataset = AtacformerMLMDataset(path_to_data, 999, 10_000)

    assert dataset is not None
    assert all([isinstance(x, tuple) for x in dataset])


@pytest.mark.skip("Too new to test")
def test_atacformer_init():
    model = Atacformer(
        10_000,  # vocab_size of 10,000 regions
    )
    assert model is not None

    input = torch.randint(0, 10_000, (32, 128))
    output = model(input)
    assert output.shape == (32, 128, 768)


@pytest.mark.skip("Too new to test")
def test_atacformer_exmodel_init(universe_file: str):
    tokenizer = AnnDataTokenizer(universe_file)
    model = AtacformerExModel(
        tokenizer=tokenizer,
    )

    # these are the defaults
    assert model._model.d_model == 768
    assert model._model.vocab_size == 2436
    assert model._model.nhead == 8
    assert model._model.num_layers == 6


@pytest.mark.skip("Too new to test")
def test_train_atacformer_ex_model(universe_file: str, data: str):
    # make tokenizer and model
    tokenizer = AnnDataTokenizer(universe_file)
    model = AtacformerExModel(
        tokenizer=tokenizer,
    )

    # curate dataset
    mask_token_id = tokenizer.mask_token_id()
    dataset = AtacformerMLMDataset(data, mask_token_id=mask_token_id, vocab_size=len(tokenizer))
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=4,
    )

    # make adapter and trainer
    adapter = MLMAdapter(model)
    trainer = L.Trainer(
        max_epochs=3,
    )
    trainer.fit(adapter, train_dataloaders=dataloader)
