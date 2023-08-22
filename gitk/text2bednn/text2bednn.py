import numpy as np
from typing import List, Union, Tuple

from .const import *

from .utils import *
from ..models import Model
from ..search import QdrantBackend, HNSWBackend
import tensorflow as tf
import matplotlib as plt


class Embed2EmbedNN(tf.keras.models.Sequential):
    def __init__(self):
        super.__init__()
        self.most_recent_train = None

    def add_layers(self,
                   input_dim: int,
                   output_dim: int,
                   units: int,
                   layers: int):

        self.add(tf.keras.layers.Dense(units, input_shape=(input_dim,), activation='relu'))

        # extra dense layers
        for i in range(layers):
            self.add(tf.keras.layers.Dense(units, input_shape=(units,), activation='relu'))

        # output
        self.add(tf.keras.layers.Dense(output_dim))

    def load_local_pretrained(self,
                              model_path: str):
        while len(self.layers) > 0:
            self.layers.pop()

        local_model = tf.keras.models.load_model(model_path)
        for layer in local_model.layers:
            self.add(layer)

    def train(self,
              training_X: np.ndarray,
              training_Y: np.ndarray,
              validating_X: np.ndarray,
              validating_Y: np.ndarray,
              opt_name: str = DEFAULT_OPTIMIZER,
              loss_func: str = DEFAULT_LOSS,
              epochs: int = DEFAULT_EPOCHES,
              batch_size: int = DEFAULT_BATCH_SIZE,
              **kwargs
              ):

        if len(self.layers) == 0:
            # dimensions of input and output
            input_dim = training_X.shape[1]
            output_dim = training_Y.shape[1]

            self.add_layers(
                input_dim=input_dim,
                output_dim=output_dim,
                units=kwargs.get("units") or DEFAULT_UNITS,
                layers=kwargs.get("layers") or DEFAULT_LAYERS
            )

        self.compile(optimizer=opt_name, loss=loss_func)
        # early stoppage to prevent over-fitting
        early_stoppage = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(epochs * 0.25))

        # record training history
        train_hist = self.fit(training_X, training_Y, epochs=epochs, batch_size=batch_size,
                              validation_data=(validating_X, validating_Y),
                              callbacks=[early_stoppage])
        self.most_recent_train = train_hist

    def embedding_to_embedding(self, input_vec: np.ndarray):
        if len(np.ndarray.shape) == 1:
            vec_dim = input_vec.shape[0]
            input_vec = input_vec.reshape((1, vec_dim))

        output_vec = self.predict(input_vec)
        if output_vec.shape[0] == 1:
            output_vec_dim = output_vec.shape[1]
            output_vec = output_vec.reshape(output_vec_dim, )
        return output_vec

    def plotting_training_hist(self):

        epoch_range = range(1, len(self.most_recent_train.history['loss']) + 1)
        train_loss = self.most_recent_train.history['loss']
        valid_loss = self.most_recent_train.history['val_loss']
        plt.plot(epoch_range, train_loss, 'r', label='Training loss')
        plt.plot(epoch_range, valid_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()


class TextToBedNNSearchInterface(object):
    def __init__(self,
                 nl2vec_model: Union[SentenceTransformer, None],
                 vec2vec_model: Union[Embed2EmbedNN, None],
                 region_set_backend: Union[QdrantBackend, HNSWBackend, None]):

        if isinstance(nl2vec_model, None):
            self.set_sentence_transformer()
        else:
            self.nl2vec = nl2vec_model

        if isinstance(vec2vec_model, None):
            self.vec2vec = Embed2EmbedNN()
        else:
            self.vec2vec = vec2vec_model

        if isinstance(region_set_backend, None):
            self.region_set_backend = QdrantBackend()
        else:
            self.region_set_backend = region_set_backend

    def set_sentence_transformer(self,
                                 st_repo: str = DEFAULT_HF_ST_MODEL):
        self.nl2vec = SentenceTransformer(st_repo)

    def nl_vec_search(self,
                      query: Union[str, np.ndarray],
                 k: int = 10) -> List:
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
        #
        return self.region_set_backend.search(search_vector, k)

    # def evaluate(self,
    #              ):

    # Todo
    # function that load a folder of bed files into the qdrant client db
    # then fit the embed2embed nn with metadata
#
# # Example of how to use the TextToBedNN Search Interface
#
# betum = BEDEmbedTUM(RSC, universe, tokenizer)
# embeddings = betum.compute_embeddings()
# T2BNNSI = TextToBedNNSearchInterface(betum, embeddings)  # ???


#
# r2v_model = Region2VecExModel(r2v_hf_repo)
# st_model = SentenceTransformer(st_hf_repo)
# bme_set = build_BedMetadataSet_from_files(bed_folder, metadata_path, r2v_model, st_model)
# assert bme_set is not None
# git add -A && git commit -m "Update comments and passed pytest"


# class TextToBedNN(Model):
#     """An Extended Model for TextToBed using a FFNN."""
#
#     def __init__(self,
#                  model_nn_path: Union[str, None] = None,
#                  model_st_repo: str = DEFAULT_HF_ST_MODEL):
#         # super().__init__(model, universe, tokenizer)
#         self.built = True
#         self._st_model = SentenceTransformer(model_st_repo)
#         if model_nn_path is not None:
#             # upload local models
#             self.from_pretrained(model_nn_path)
#         else:
#             # create an empty Sequential
#             self._nn_model = tf.keras.models.Sequential()
#             self.built = False  # model is not built
#         # initiation
#
#     def __repr__(self):
#         print("To be completed")
#
#     def init_from_huggingface(self):
#         # to be finished
#         print("To be completed")
#
#     def from_pretrained(self, model_nn_path: str):
#         """
#         load pretrained model from local file
#
#         :param model_nn_path: the local path of saved model
#         :return:
#         """
#         self._nn_model = tf.keras.models.load_model(model_nn_path)
#
#     def train(self, dataset: BedMetadataEmbeddingSet,
#               loss_func: str = DEFAULT_LOSS,
#               epochs: int = DEFAULT_EPOCHES,
#               batch_size: int = DEFAULT_BATCH_SIZE,
#               units: int = DEFAULT_UNITS,
#               layers: int = DEFAULT_LAYERS,
#               plotting: bool = False):
#         """
#         train the feedforward neural network model with a given BedMetadataEmbeddingSet
#
#         :param dataset: a BedMetadataEmbeddingSet
#         :param loss_func: loss function for training
#         :param epochs: number of training epochs
#         :param batch_size: training batch size
#         :param units: number of neurons in a dense layer
#         :param layers: number of extra layers
#         :param plotting: whether to plot the training and velidation loss by epochs
#         :return:
#         """
#
#         # get training and validating dataset from BedMetadataEmbeddingSet
#         training_X, training_Y = dataset.generate_data("training")
#         validating_X, validating_Y = dataset.generate_data("validating")
#
#         # if the model is empty
#         if not self.built:
#             print("Build new sequential model")
#
#             # dimensions of input and output
#             input_dims = training_X[0].shape
#             output_dims = training_Y[0].shape
#
#             # the dense layer that connected to input
#             self._nn_model.add(tf.keras.layers.Dense(units, input_shape=input_dims, activation='relu'))
#
#             # extra dense layers
#             for i in range(layers):
#                 self._nn_model.add(tf.keras.layers.Dense(units, input_shape=(units,), activation='relu'))
#
#             # output
#             self._nn_model.add(tf.keras.layers.Dense(output_dims[0]))
#
#             self._nn_model.compile(optimizer="adam", loss=loss_func)
#
#             # model is no longer empty
#             self.built = True
#
#         # early stoppage to prevent overfitting
#         early_stoppage = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(epochs * 0.25))
#
#         # record training history
#         train_hist = self._nn_model.fit(training_X, training_Y, epochs=epochs, batch_size=batch_size,
#                                         validation_data=(validating_X, validating_Y),
#                                         callbacks=[early_stoppage])
#
#         # plot the training history: training and validation loss by epoch
#         if plotting:
#             epoch_range = range(1, len(train_hist.history['loss']) + 1)
#             train_loss = train_hist.history['loss']
#             valid_loss = train_hist.history['val_loss']
#             plt.plot(epoch_range, train_loss, 'r', label='Training loss')
#             plt.plot(epoch_range, valid_loss, 'b', label='Validation loss')
#             plt.title('Training and validation loss')
#             plt.legend()
#             plt.show()
#
#     def forward(self, query: str) -> np.ndarray:
#         """
#         map a query string to an embedding vector of bed file
#
#         :param query: a string
#         :return:
#         """
#
#         # input string -> embedded by sentence transformer -> FNN -> vector
#
#         # encode the query string to vector by sentence transformer
#         query_embedding = self._st_model.encode(query)
#
#         # reshape the encoding vector
#         # because the model takes input with shape of (n, <vector dimension>)
#         # and the shape of encoding vector is (<vector dimension>, )
#         vec_dim = query_embedding.shape[0]
#
#         # map the encoding vector to bedfile embedding vector by feedforward neural network
#         output_vec = self._nn_model.predict(query_embedding.reshape((1, vec_dim)))
#
#         # reshape the output from (1, <region2vec embedding dimension>) to (<~ dimension>, 1)
#         output_vec_dim = output_vec.shape[1]
#         return output_vec.reshape(output_vec_dim, )
#
#
