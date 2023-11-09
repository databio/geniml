import logging
import os
from typing import Union, List

import torch
import torch.nn as nn
import scanpy as sc
from rich.progress import Progress
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from yaml import safe_dump, safe_load

from .const import (
    MODULE_NAME,
    MODEL_FILE_NAME,
    UNIVERSE_FILE_NAME,
    CONFIG_FILE_NAME,
    DEFAULT_EPOCHS,
    DEFAULT_LABEL_KEY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOSS_FN,
    DEFAULT_OPTIMIZER,
    DEFAULT_TEST_TRAIN_SPLIT,
)
from .utils import SingleCellClassificationDataset
from ..region2vec.const import DEFAULT_EMBEDDING_SIZE
from ..region2vec.main import Region2Vec
from ..tokenization.main import ITTokenizer

_LOGGER = logging.getLogger(MODULE_NAME)


class SingleCellTypeClassifier(nn.Module):
    def __init__(self, region2vec: Region2Vec, num_classes: int, freeze_r2v: bool = False):
        """
        Initialize the SingleCellTypeClassifier.

        :param Union[Region2Vec, str] region2vec: Either a Region2Vec instance or a path to a huggingface model.
        :param int num_classes: Number of classes to classify.
        :param bool freeze_r2v: Whether or not to freeze the Region2Vec model.
        """
        super().__init__()

        self.region2vec: Region2Vec = region2vec

        # freeze the weights of the Region2Vec model
        if freeze_r2v:
            for param in self.region2vec.parameters():
                param.requires_grad = False

        self.num_classes = num_classes
        self.output_layer = nn.Linear(self.region2vec.embedding_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.region2vec(x)
        x = x.sum(dim=1)
        x = self.output_layer(x)
        return x


class SingleCellTypeClassifierExModel:
    def __init__(
        self,
        model_path: str = None,
        tokenizer: ITTokenizer = None,
        region2vec: Union[Region2Vec, str] = None,
        num_classes: int = None,
        device: str = None,
        freeze_r2v: bool = False,
        **kwargs,
    ):
        """
        Initialize the SingleCellTypeClassifierExModel from a huggingface model or from scratch.
        """

        # there really are two models here, the Region2Vec model and the SingleCellTypeClassifier model
        # which holds the Region2Vec model. The Region2Vec model ideally is pre-trained, but can be trained
        # from scratch if a tokenizer is passed. The SingleCellTypeClassifier model is always trained from scratch.
        self._model = SingleCellTypeClassifier
        self.region2vec: Region2Vec
        self.device = device
        self.trained = False
        self._label_mapping: dict

        # check for completely blank initialization
        if all(
            [
                model_path is None,
                tokenizer is None,
                region2vec is None,
            ]
        ):
            pass
        # try to load the model from huggingface
        elif model_path is not None:
            self._init_from_huggingface(model_path)
            self.trained = True
        else:
            self._init_model(region2vec, num_classes, tokenizer, freeze_r2v, **kwargs)
            self._label_mapping = {}

    def _load_local_region2vec_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load a Region2Vec model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        self._model_path = model_path
        self._universe_path = vocab_path

        # init the tokenizer - only one option for now
        self.tokenizer = ITTokenizer(vocab_path, verbose=False)

        # load the model state dict (weights)
        params = torch.load(model_path)

        # get the model config (vocab size, embedding size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self.region2vec = Region2Vec(
            config["vocab_size"],
            embedding_dim=config["embedding_size"],
        )
        self.region2vec.load_state_dict(params)

    def _init_region2vec_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the Region2Vec model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those. This
        includes both the actual model and the tokenizer.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param kwargs: Additional keyword arguments to pass to the hf download function.
        """
        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self._load_local_region2vec_model(model_file_path, universe_path, config_path)

    def _init_tokenizer(self, tokenizer: Union[ITTokenizer, str]):
        """
        Initialize the tokenizer.

        :param Union[ITTokenizer, str] tokenizer: Either a tokenizer instance, a huggingface repo, or a path to a tokenizer.
        """
        # they didn't pass anything valid, so we try to build
        # a region2vec model from scratch, please pass a tokenizer :(
        if tokenizer is None:
            raise ValueError(
                "Can't build a Region2Vec model from scratch without a tokenizer. A vobab size is needed!"
            )
        elif isinstance(tokenizer, str):
            # is path to a local vocab?
            if os.path.exists(tokenizer):
                self.tokenizer = ITTokenizer(tokenizer, verbose=False)
            else:
                # assume its a huggingface tokenizer, try to load it
                self.tokenizer = ITTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, ITTokenizer):
            # ideal case - they passed a tokenizer instance
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                "Invalid tokenizer passed. Must be either a path to a tokenizer, a huggingface tokenizer, or a tokenizer instance."
            )

    def _init_model(self, region2vec, num_classes, tokenizer, freeze_r2v: bool = False, **kwargs):
        """
        Initialize a new model from scratch. Ideally, someone is passing in a Region2Vec instance,
        but in theory we can build one from scratch if we have a tokenizer and a `num_classes` parameter.
        In the "build from scratch" scenario, someone is probably training a new model from scratch on some new data.

        The most important aspect is initializing the inner Region2Vec model. For this, the order of operations
        is as follows:

        1. If a Region2Vec instance is passed, use that.
        2. If a path to a local Region2Vec model is passed, use that.
        3. If a path to a huggingface Region2Vec model is passed, use that.
        4. Nothing was passed? Try from scratch... if a tokenizer is passed, build a Region2Vec model from scratch.
        5. If no tokenizer was passed, raise an error, because we need a vocab size to build a Region2Vec model.

        :param Union[Region2Vec, str] region2vec: Either a Region2Vec instance or a path to a huggingface model.
        :param int num_classes: Number of classes to classify. This is **required**.
        :param ITTokenizer tokenizer: The tokenizer to use. This is **required** if building a Region2Vec model from scratch.
        :param bool freeze_r2v: Whether or not to freeze the Region2Vec model.
        :param kwargs: Additional keyword arguments to pass to the Region2Vec model.
        """
        # first init the tokenizer they passed in
        self._init_tokenizer(tokenizer)

        if isinstance(region2vec, Region2Vec):
            # ideal case - they passed a Region2Vec instance
            self.region2vec = region2vec

        elif isinstance(region2vec, str):
            # is it a local model?
            if os.path.exists(region2vec):
                self._load_local_region2vec_model(
                    os.path.join(region2vec, MODEL_FILE_NAME),
                    os.path.join(region2vec, UNIVERSE_FILE_NAME),
                    os.path.join(region2vec, CONFIG_FILE_NAME),
                )
            else:
                # assume its a huggingface model, try to load it
                self._init_region2vec_from_huggingface(region2vec)

        else:
            self.region2vec = Region2Vec(
                len(self.tokenizer),
                embedding_dim=kwargs.get("embedding_dim") or DEFAULT_EMBEDDING_SIZE,
            )

        assert (
            len(self.tokenizer) == self.region2vec.vocab_size
        ), "Tokenizer and Region2Vec vocab size mismatch. Are you sure they are compatible?"

        if num_classes is None:
            raise ValueError("Must pass a number of classes to build classifier!")

        self.num_classes = num_classes

        # build the model, finally using the region2vec model
        self._model = SingleCellTypeClassifier(self.region2vec, num_classes, freeze_r2v)

    def _load_local_model(self, model_path: str, vocab_path: str, config_path: str):
        """
        Load the model from a checkpoint.

        :param str model_path: Path to the model checkpoint.
        :param str vocab_path: Path to the vocabulary file.
        """
        # init the tokenizer - only one option for now
        self.tokenizer = ITTokenizer(vocab_path, verbose=False)

        # load the model state dict (weights)
        params = torch.load(model_path)

        # get the model config (vocab size, embedding size)
        with open(config_path, "r") as f:
            config = safe_load(f)

        self.region2vec = Region2Vec(
            config["vocab_size"],
            embedding_dim=config["embedding_size"],
        )

        self._label_mapping = config["label_mapping"]
        self.num_classes = config["num_classes"]

        self._model = SingleCellTypeClassifier(self.region2vec, config["num_classes"])
        self._model.load_state_dict(params)

    def _init_from_huggingface(
        self,
        model_path: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
        **kwargs,
    ):
        """
        Initialize the model from a huggingface model. This uses the model path
        to download the necessary files and then "build itself up" from those. This
        includes both the actual model and the tokenizer.

        :param str model_path: Path to the pre-trained model on huggingface.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param kwargs: Additional keyword arguments to pass to the hf download function.
        """
        model_file_path = hf_hub_download(model_path, model_file_name, **kwargs)
        universe_path = hf_hub_download(model_path, universe_file_name, **kwargs)
        config_path = hf_hub_download(model_path, config_file_name, **kwargs)

        self._load_local_model(model_file_path, universe_path, config_path)

    @classmethod
    def from_pretrained(
        cls,
        path_to_files: str,
        model_file_name: str = MODEL_FILE_NAME,
        universe_file_name: str = UNIVERSE_FILE_NAME,
        config_file_name: str = CONFIG_FILE_NAME,
    ) -> "SingleCellTypeClassifierExModel":
        """
        Load the model from a set of files that were exported using the export function.

        :param str path_to_files: Path to the directory containing the files.
        :param str model_file_name: Name of the model file.
        :param str universe_file_name: Name of the universe file.
        :param str config_file_name: Name of the config file.

        :return: The loaded model.
        """
        model_file_path = os.path.join(path_to_files, model_file_name)
        universe_file_path = os.path.join(path_to_files, universe_file_name)
        config_file_path = os.path.join(path_to_files, config_file_name)

        instance = cls()
        instance._load_local_model(model_file_path, universe_file_path, config_file_path)
        instance.trained = True

        return instance

    def _validate_data(self, data: Union[sc.AnnData, str], label_key: str) -> sc.AnnData:
        """
        Validate the data passed to the training function.

        :param Union[sc.AnnData, str] data: Either a scanpy AnnData object or a path to a h5ad file.
        :param str label_key: The key in the `.obs` attribute that contains the cell type labels.

        :return: The scanpy AnnData object.
        """
        if isinstance(data, str):
            data = sc.read_h5ad(data)
        elif not isinstance(data, sc.AnnData):
            raise ValueError(
                "Data must be either a scanpy AnnData object or a path to a h5ad file."
            )
        assert label_key in data.obs.columns, f"Label key {label_key} not found in data."
        return data

    def class_to_label(self, class_id: int) -> str:
        """
        Convert a class id to a label.

        :param int class_id: The class id to convert.

        :return: The label.
        """
        try:
            return self._label_mapping[class_id]
        except AttributeError:
            raise RuntimeError("Model has not label mapping, are you sure it is trained?")
        except KeyError:
            raise ValueError(f"Class id {class_id} not found in label mapping.")

    def freeze_r2v(self):
        """
        Freeze the weights of the Region2Vec model.
        """
        for param in self.region2vec.parameters():
            param.requires_grad = False

    def train(
        self,
        data: Union[sc.AnnData, str],
        label_key: str = DEFAULT_LABEL_KEY,
        epochs: int = DEFAULT_EPOCHS,
        loss_fn: nn.Module = DEFAULT_LOSS_FN,
        optimizer: torch.optim.Optimizer = DEFAULT_OPTIMIZER,
        batch_size: int = DEFAULT_BATCH_SIZE,
        optimizer_kwargs: dict = {},
        test_train_split: float = DEFAULT_TEST_TRAIN_SPLIT,
        seed: any = 42,
        learning_rate_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: Union[List, str] = None,
    ):
        """
        Train the model. The training loop assumes your data is a scanpy AnnData object,
        with the cell type labels in the `.obs` attribute. The cell type labels are
        expected to be in the column specified by `label_key`.

        :param Union[sc.AnnData, str] data: Either a scanpy AnnData object or a path to a h5ad file.
        :param str label_key: The key in the `.obs` attribute that contains the cell type labels.
        :param int epochs: Number of epochs to train for.
        :param nn.Module loss_fn: Loss function to use.
        :param torch.optim.Optimizer optimizer: Optimizer to use.
        :param int batch_size: Batch size to use.
        :param dict optimizer_kwargs: Additional keyword arguments to pass to the optimizer.
        :param float test_train_split: Fraction of data to use for training (defaults to 0.8).
        """
        # validate the data
        data = self._validate_data(data, label_key)

        # convert labels to integers
        data.obs[f"{label_key}_code"] = data.obs[label_key].astype("category").cat.codes

        # get mapping from integer to label
        self._label_mapping = dict(
            enumerate(data.obs[label_key].astype("category").cat.categories)
        )

        # split the data into train and test
        train, test, _, _ = train_test_split(
            data, data.obs[label_key], train_size=test_train_split, random_state=seed
        )
        del data

        # convert to better datatypes
        # the labels are stored in the `.obs` attribute
        train = train.to_memory()
        test = test.to_memory()

        # create the datasets
        train_dataset = SingleCellClassificationDataset(train, label_key=f"{label_key}_code")
        test_dataset = SingleCellClassificationDataset(test, label_key=f"{label_key}_code")

        # create the dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True
        )

        losses = []

        # move the model to the device
        if isinstance(device, list):
            self._model = nn.DataParallel(self._model, device_ids=device)
            self._model.to(device[0])
        elif isinstance(device, str):
            self._model.to(device)

        # tensor device
        if isinstance(device, list):
            tensor_device = device[0]
        else:
            tensor_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # move the loss function to the device
        loss_fn = loss_fn()

        # create the optimizer
        optimizer = optimizer(self._model.parameters(), **optimizer_kwargs)

        # init scheduler if passed
        if learning_rate_scheduler is not None:
            learning_rate_scheduler = learning_rate_scheduler(optimizer)

        with Progress() as progress_bar:
            epoch_tid = progress_bar.add_task("Epochs", total=epochs)
            batches_tid = progress_bar.add_task("Batches", total=len(train_dataloader))
            for epoch in range(epochs):
                for _, batch in enumerate(train_dataloader):
                    x, y = batch

                    # zero the gradients
                    optimizer.zero_grad()

                    # move the data to the device
                    x = x.to(tensor_device)
                    y = y.to(tensor_device)

                    # forward pass
                    y_pred = self._model(x)

                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    losses.append(loss.item())

                    # update the weights
                    optimizer.step()

                    if learning_rate_scheduler is not None:
                        learning_rate_scheduler.step()

                    # update the progress bar
                    progress_bar.update(batches_tid, advance=1)

            # update the progress bar
            progress_bar.update(epoch_tid, advance=1)

            _LOGGER.info("Finished training.")

        self.trained = True

        # run testing
        # TODO: add testing

        return losses

    def predict(self, data: Union[sc.AnnData, str], label_key: str = DEFAULT_LABEL_KEY):
        """
        Predict the cell types of the given data.

        :param Union[sc.AnnData, str] data: Either a scanpy AnnData object or a path to a h5ad file.
        :param str label_key: The key in the `.obs` attribute that contains the cell type labels.
        """
        if isinstance(data, str):
            data = sc.read_h5ad(data)
        elif not isinstance(data, sc.AnnData):
            raise ValueError(
                "Data must be either a scanpy AnnData object or a path to a h5ad file."
            )

        tokens = self.tokenizer.tokenize(data)
        tokens = torch.tensor(tokens)

        predictions = self._model(tokens)
        predictions = torch.argmax(predictions, dim=1)

        # convert the predictions to labels
        predictions = [self._label_mapping[p.item()] for p in predictions]

        return predictions

    def export(
        self,
        path: str,
        checkpoint_file: str = MODEL_FILE_NAME,
        universe_file: str = UNIVERSE_FILE_NAME,
        config_file: str = CONFIG_FILE_NAME,
    ):
        """
        Function to facilitate exporting the model in a way that can
        be directly uploaded to huggingface. This exports the model
        weights and the vocabulary.

        :param str path: Path to export the model to.
        """
        # make sure the path exists
        if not os.path.exists(path):
            os.makedirs(path)

        # export the model weights
        torch.save(self._model.state_dict(), os.path.join(path, checkpoint_file))

        # export the vocabulary
        with open(os.path.join(path, universe_file), "a") as f:
            for region in self.tokenizer.universe.regions:
                f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

        # export the config (vocab size, embedding size)
        config = {
            "vocab_size": len(self.tokenizer),
            "embedding_size": self._model.region2vec.embedding_dim,
            "num_classes": self._model.num_classes,
            "label_mapping": self._label_mapping or {},
        }

        with open(os.path.join(path, config_file), "w") as f:
            safe_dump(config, f)
