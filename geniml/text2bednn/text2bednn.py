import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

from ..search.backends import HNSWBackend, QdrantBackend
from .const import *
from .utils import *


class Vec2VecFNN(tf.keras.models.Sequential):
    """
    A feedforward neural network that maps embedding vectors or region sets metadata
    to the embedding vectors of region sets
    """

    def __init__(self, model_path: str = None):
        """
        initialization

        :param str model_path: Path to the pre-trained model on huggingface.
        """
        super().__init__()
        # initiate a Sequential model from keras
        if model_path is not None:
            self._load_from_huggingface(model_path)

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

    def load(self, model_path: str):
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

    def _load_from_huggingface(
        self,
        model_repo: str,
        model_file_name: str = MODEL_FILE_NAME,
        **kwargs,
    ):
        model_path = hf_hub_download(model_repo, model_file_name, **kwargs)
        self.load(model_path)

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

        if isinstance(nl2vec_model, type(None)):
            # default SentenceTransformer model
            self.set_sentence_transformer()
        else:
            self.nl2vec = nl2vec_model

        if isinstance(vec2vec_model, type(None)):
            # init an empty Vec2VecFNN model if input is None
            self.vec2vec = Vec2VecFNN()
        elif isinstance(vec2vec_model, str):
            self.vec2vec = Vec2VecFNN()
            self.vec2vec.load_local_pretrained(vec2vec_model)
        elif isinstance(vec2vec_model, Vec2VecFNN):
            self.vec2vec = vec2vec_model
        else:
            raise TypeError(
                "vec2vec_model must be either a path to a pretrained model or a Vec2VecFNN model"
            )

        if isinstance(search_backend, type(None)):
            # init a default HNSWBackend if input is None
            self.search_backend = HNSWBackend()
        else:
            self.search_backend = search_backend

    def set_sentence_transformer(self, st_repo: str = DEFAULT_HF_ST_MODEL):
        """
        With a given huggingface repo, set the nl2vec model as a sentence transformer

        :param st_repo: the hugging face repository of sentence transformer
        see https://huggingface.co/sentence-transformers
        :return:
        """
        _LOGGER.info(f"Setting sentence transformer model {st_repo}")
        self.nl2vec = SentenceTransformer(st_repo)

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
