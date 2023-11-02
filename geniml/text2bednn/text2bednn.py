import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

from ..search.backends import HNSWBackend, QdrantBackend
from .const import *
from .utils import *
import math

import torch

from torch.nn import Sequential, Linear, ReLU, CosineEmbeddingLoss, CosineSimilarity

import torch.nn.functional as F

_LOGGER = logging.getLogger(MODULE_NAME)


class Vec2VecFNNtorch(Sequential):
    def __init__(self, trained: bool = False):
        """

        """
        super(Vec2VecFNNtorch, self).__init__()
        self.most_recent_train = None
        self.optimizer = None
        self.loss = None
        self.compiling_info = {}

    def add_layers(
            self, input_dim: int, output_dim: int, num_units: Union[int, List[int]], num_extra_hidden_layers: int
    ):
        """
        Add layers to an empty nn.Sequential model
        """
        if isinstance(num_units, list):
            num_units = [num_units] * (1 + num_extra_hidden_layers)

        if len(num_units) != 1 + num_extra_hidden_layers:
            _LOGGER.error("ValueError: list of units number does not match number of layers")

        current_layer_units_num = num_units.pop(0)
        layers_list = [Linear(in_features=input_dim, out_features=current_layer_units_num), ReLU()]
        previous_layer_units_num = current_layer_units_num
        for i in range(num_extra_hidden_layers):
            current_layer_units_num = num_units.pop(0)
            layers_list.append(Linear(in_features=previous_layer_units_num, out_features=current_layer_units_num))
            layers_list.append(ReLU())
            previous_layer_units_num = current_layer_units_num
        layers_list.append(Linear(in_features=previous_layer_units_num, out_features=output_dim))
        super(Vec2VecFNNtorch, self).__init__(*layers_list)

    def load_from_disk(self, model_path: str):
        self.load_state_dict(torch.load(model_path))

    def embedding_to_embedding(self, input_vecs: np.ndarray) -> np.ndarray:
        """
        predict the region set embedding from embedding of natural language strings

        :param input_vecs:
        :return:
        """
        # pytorch tensor's default dtype is float 32
        if not isinstance(input_vecs.dtype, type(np.dtype("float32"))):
            input_vecs = input_vecs.astype(np.float32)
        return self(torch.from_numpy(input_vecs)).detach().numpy()

    def compile(self, optimizer: str, loss: str, learning_rate: float):
        compiling_dict = {}
        # set optimizer
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)

        else:
            _LOGGER.error("ValueError: Please give a valid name of optimizer")


        # set loss function
        if loss == "cosine_embedding_loss":
            self.loss = CosineEmbeddingLoss()
        elif loss == "cosine_similarity":
            self.loss == CosineSimilarity()
        else:
            _LOGGER.error("ValueError: Please give a valid name of loss function")

        compiling_dict["optimizer"] = optimizer
        compiling_dict["loss"] = loss

    def train_with_vecs(
            self,
            training_X: np.ndarray,
            training_Y: np.ndarray,
            validating_data: Union[Tuple[np.ndarray, np.ndarray], None] = None,
            save_best: bool = False,
            folder_path: Union[str, None] = None,
            early_stop: bool = True,
            patience: float = DEFAULT_PATIENCE,
            opt_name: str = DEFAULT_OPTIMIZER_NAME,
            loss_func: str = DEFAULT_LOSS_NAME,
            num_epochs: int = DEFAULT_NUM_EPOCHS,
            batch_size: int = DEFAULT_BATCH_SIZE,
            learning_rate: float = DEFAULT_LEARNING_RATE,
            **kwargs,
    ):
        if self.__len__() == 0:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.add_layers(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=kwargs.get("num_units") or DEFAULT_NUM_UNITS,
                num_extra_hidden_layers=kwargs.get("num_extra_hidden_layers")
                                        or DEFAULT_NUM_EXTRA_HIDDEN_LAYERS,
            )

        if validating_data is not None:
            validating_X, validating_Y = validating_data
            validating_data = arrays_to_torch_dataloader(validating_X, validating_Y, shuffle=False)
        elif save_best or early_stop:
            _LOGGER.error("ValueError: Validating data is not provided")
        if save_best and folder_path is None:
            _LOGGER.error("ValueError: Path to folder where the best performance model will be saved is required")

        self.compile(optimizer=opt_name, loss=loss_func, learning_rate=learning_rate)
        training_data = arrays_to_torch_dataloader(training_X, training_Y, batch_size)
        best_val_loss = 1_000_000.

        patience_count = 0
        for epoch in range(num_epochs):
            self.train(True)
            avg_loss = self.train_one_epoch(training_data)

            self.eval()

            if validating_data is not None:
                running_val_loss = 0.0
                with torch.no_grad():
                    for i, (val_x, val_y) in enumerate(validating_data):
                        val_output = self(val_x)
                        val_loss = self.loss(val_output, val_y)
                        running_val_loss += val_loss

                avg_val_loss = running_val_loss / (i+1)
                # print
                _LOGGER.info(f"EPOCH {epoch + 1}: loss: -{avg_loss} - val_loss: -{avg_val_loss}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if save_best:
                        model_path = os.path.join(folder_path, f"model_{epoch}")
                        torch.save(self.state_dict(), model_path)

                if early_stop:
                    if avg_val_loss > avg_loss:
                        patience_count += 1
                    if patience_count > int(math.ceil(patience * num_epochs)):
                        break

            else:
                _LOGGER.info(f"EPOCH {epoch + 1}: loss: -{avg_loss}")

    def train_one_epoch(self, training_data):
        epoch_loss = 0.

        for i, (x, y) in enumerate(training_data):
            self.optimizer.zero_grad()
            outputs = self(y)

            batch_loss = self.loss(outputs, y, )
            self.loss.backward()

            self.optimizer.step()

            epoch_loss += batch_loss.item()

        return epoch_loss / (i + 1)

    def calc_loss(self, outputs: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.float:
        if self.compiling_info == {}:
            _LOGGER.error("ValueError: Please compile the model first")

        if self.compiling_info["loss"] == "cosine_similarity":
            return 1 - self.loss(outputs, y)

        if self.compiling_info["loss"] == "cosine_embedding_loss":
            target = kwargs.get("target")
            return self.loss(outputs, y, target)


class Vec2VecFNN(tf.keras.models.Sequential):
    """
    A feedforward neural network that maps embedding vectors or region sets metadata
    to the embedding vectors of region sets
    """

    def __init__(self, model_path: str = None):
        """
        Initializate Vec2VecFNN.

        :param str model_path: Path to the pre-trained model on huggingface.
        """
        super().__init__()
        # initiate a Sequential model from keras
        if model_path is not None:
            if os.path.exists(model_path):
                # load from disk
                self.load_from_disk(model_path)
            else:
                # load from hugging face
                self.load_from_huggingface(model_path)

        # most recent training history
        self.most_recent_train = None

    def add_layers(
        self, input_dim: int, output_dim: int, num_units: int, num_extra_hidden_layers: int
    ):
        """
        Add layers to an empty Sequential model

        :param input_dim: dimension of input vector
        :param output_dim: dimension of output vector
        :param num_units: number of units in dense layer
        :param num_extra_hidden_layers: number of extra hidden layers
        :return:
        """
        # the dense layer that accept the input
        self.add(tf.keras.layers.Dense(num_units, input_shape=(input_dim,), activation="relu"))

        # extra dense layers
        for i in range(num_extra_hidden_layers):
            self.add(tf.keras.layers.Dense(num_units, input_shape=(num_units,), activation="relu"))

        # output
        self.add(tf.keras.layers.Dense(output_dim))

    def load_from_disk(self, model_path: str):
        """
        load pretrained model from disk

        :param model_path: path where the model is saved
        :return:
        """

        # empty the layers if current model is not empty
        while len(self.layers) > 0:
            self.layers.pop()

        # https://stackoverflow.com/questions/63068639/valueerror-unknown-layer-functional
        local_model = tf.keras.models.load_model(
            model_path, custom_objects={"Vec2VecFNN": tf.keras.models.Sequential()}
        )
        # add layers from pretrained model
        for layer in local_model.layers:
            self.add(layer)

    def load_from_huggingface(
        self,
        model_repo: str,
        model_file_name: str = DEFAULT_VEC2VEC_MODEL_FILE_NAME,
        **kwargs,
    ):
        """
        Download pretrained model from huggingface

        :param str model: The name of the model to download (this is the same as the repo name).
        :param str model_file_name: The name of the model file - this should almost never be changed.
        """
        model_path = hf_hub_download(model_repo, model_file_name, **kwargs)
        self.load_from_disk(model_path)

    def train(
        self,
        training_X: np.ndarray,
        training_Y: np.ndarray,
        validating_data: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        opt_name: str = DEFAULT_OPTIMIZER_NAME,
        loss_func: str = DEFAULT_LOSS_NAME,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        patience: float = DEFAULT_PATIENCE,
        **kwargs,
    ):
        """
        Fit the feedforward neural network

        :param training_X: embedding vectors of metadata, np.ndarray with shape of (n, <dim>)
        :param training_Y: embedding vectors of region set, np.ndarray with shape of (n, <dim>)
        :param validating_data: validating data, which contains validating X and validating Y
        :param opt_name: name of optimizer
        :param loss_func: name of loss function
        :param num_epochs: number of training epoches
        :param batch_size: size of batch for training
        :param learning_rate: learning rate of optimizer
        :param patience: the percentage of epoches in which if no validation loss improvement,
        the training will be stopped
        :param kwargs: see units and layers in add_layers()
        :return:
        """

        # if current model is empty, add layers
        if len(self.layers) == 0:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.add_layers(
                input_dim=input_dim,
                output_dim=output_dim,
                num_units=kwargs.get("num_units") or DEFAULT_NUM_UNITS,
                num_extra_hidden_layers=kwargs.get("num_extra_hidden_layers")
                or DEFAULT_NUM_EXTRA_HIDDEN_LAYERS,
            )

        # compile the model
        self.compile(optimizer=opt_name, loss=loss_func)
        # set the learning rate
        tf.keras.backend.set_value(self.optimizer.learning_rate, learning_rate)
        # if there is validating data, set early stoppage to prevent over-fitting
        callbacks = None
        if validating_data:
            early_stoppage = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(num_epochs * patience)
            )
            callbacks = [early_stoppage]

        # record training history
        train_hist = self.fit(
            training_X,
            training_Y,
            epochs=num_epochs,
            batch_size=batch_size,
            validation_data=validating_data,
            callbacks=callbacks,
        )
        self.most_recent_train = train_hist

    def embedding_to_embedding(self, input_vec: np.ndarray):
        """
        predict the region set embedding from embedding of natural language strings
        :param input_vec:
        :return:
        """
        # the network only accept input vectors in shape of (n, <input dim>)
        # so if the input np.ndarray has shape (<input dim>,)
        # it needs reshaping
        if len(input_vec.shape) == 1:
            vec_dim = input_vec.shape[0]
            input_vec = input_vec.reshape((1, vec_dim))

        # reshape output vector if the input np.ndarray has shape (<input dim>,)
        output_vec = self.predict(input_vec)
        if output_vec.shape[0] == 1:
            output_vec_dim = output_vec.shape[1]
            output_vec = output_vec.reshape(
                output_vec_dim,
            )
        return output_vec

    def plot_training_hist(self, save_path: Union[str, None] = None):
        """
        plot the training & validating loss of the most recent training
        :return:
        """

        epoch_range = range(1, len(self.most_recent_train.history["loss"]) + 1)
        train_loss = self.most_recent_train.history["loss"]
        valid_loss = self.most_recent_train.history["val_loss"]
        plt.plot(epoch_range, train_loss, "r", label="Training loss")
        plt.plot(epoch_range, valid_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, "train_hist.png"))
        else:
            plt.show()


class Text2BEDSearchInterface(object):
    """
    search backend interface
    """

    def __init__(
        self,
        nl2vec_model: Union[SentenceTransformer, None],
        vec2vec_model: Union[Vec2VecFNN, str, None],
        search_backend: Union[QdrantBackend, HNSWBackend, None],
    ):
        """
        initiate the search interface

        :param nl2vec_model: model that embed natural language to vectors
        :param vec2vec_model: model that map natural language embedding vectors to region set embedding vectors
        :param search_backend: search backend that can store vectors and perform KNN search
        """
        # load the natural language encoder model
        if isinstance(nl2vec_model, type(None)):
            # default SentenceTransformer model
            self.set_sentence_transformer()
        else:
            self.nl2vec = nl2vec_model

        # load the vec2vec model
        if isinstance(vec2vec_model, Vec2VecFNN):
            self.vec2vec = vec2vec_model
        elif isinstance(vec2vec_model, str):
            self.set_vec2vec(vec2vec_model)

        # init search backend
        if isinstance(search_backend, type(None)):
            # init a default HNSWBackend if input is None
            self.search_backend = HNSWBackend()
        else:
            self.search_backend = search_backend

    def set_vec2vec(self, model_name: str):
        """
        With a given model_path or huggingface repo, set the vec2vec model

        :param model_name: the path where the model file is saved, or the hugging face repo
        """
        self.vec2vec = Vec2VecFNN(model_name)

    def set_sentence_transformer(self, hf_repo: str = DEFAULT_HF_ST_MODEL):
        """
        With a given huggingface repo, set the nl2vec model as a sentence transformer

        :param hf_repo: the hugging face repository of sentence transformer
        see https://huggingface.co/sentence-transformers
        :return:
        """
        _LOGGER.info(f"Setting sentence transformer model {hf_repo}")
        self.nl2vec = SentenceTransformer(hf_repo)

    def nl_vec_search(
        self, query: Union[str, np.ndarray], k: int = 10
    ) -> Tuple[Union[List[int], List[List[int]]], Union[List[float], List[List[float]]]]:
        """
        Given an input natural language, suggest region sets

        :param query: searching input string
        :param k: number of results (nearst neighbor in vectors)
        :return: a list of Qdrant Client search results
        """

        # first, get the embedding of the query string
        if isinstance(query, str):
            query = self.nl2vec.encode(query)
        search_vector = self.vec2vec.embedding_to_embedding(query)
        # perform the KNN search among vectors stored in backend
        return self.search_backend.search(search_vector, k)

    def __repr__(self):
        return f"Text2BEDSearchInterface(nl2vec_model={self.nl2vec}, vec2vec_model={self.vec2vec}, search_backend={self.search_backend})"
