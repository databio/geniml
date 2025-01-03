import lightning as L
import pytest
from torch.utils.data import DataLoader

from geniml.scembed.main import ScEmbed
from geniml.scembed.utils import BatchCorrectionDataset, BCBatchCollator

# from geniml.training.adapters import AdversarialBatchCorrectionAdapter


@pytest.fixture
def universe_file():
    return "tests/data/universe_mlm.bed"


@pytest.fixture
def data():
    return "tests/data/gtok_sample/"


@pytest.mark.skip("Not implemented")
class TestAtacFormer:
    def test_atacformer_dataset(self, data: str):
        dataset = BatchCorrectionDataset(
            [data, data]
        )  # just do the same thing twice, it doesnt matter this is just a test

        assert dataset is not None
        assert all([isinstance(x, tuple) for x in dataset])

    @pytest.mark.skip(reason="This test uses a pretrained model and is not suitable for CI")
    def test_adapter_init(
        self,
    ):
        model = ScEmbed("databio/r2v-luecken2021-hg38-v2")
        adapter = AdversarialBatchCorrectionAdapter(
            model=model, mode="adversary", num_batches=2, grad_rev_alpha=1.0
        )

        assert adapter is not None

    @pytest.mark.skip(reason="This test uses a pretrained model and is not suitable for CI")
    def test_train_with_adapter(self, universe_file: str, data: str):
        # get the model
        model = ScEmbed("databio/r2v-luecken2021-hg38-v2")

        # make adapter
        adapter = AdversarialBatchCorrectionAdapter(
            model=model, mode="adversary", num_batches=2, grad_rev_alpha=1.0
        )

        # make dataset
        collator = BCBatchCollator(model.tokenizer.padding_token_id())
        dataset = BatchCorrectionDataset(
            [data, data]
        )  # just do the same thing twice, it doesnt matter this is just a test

        # make dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)

        # make trainer
        trainer = L.Trainer(max_epochs=3)

        # train in the adversarial mode
        trainer.fit(adapter, dataloader)

        # switch to the batch correction mode
        adapter.set_mode("batch_correction")

        # train in the batch correction mode
        trainer.fit(adapter, dataloader)
