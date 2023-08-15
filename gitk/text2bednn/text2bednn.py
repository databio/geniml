import numpy as np
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
from .const import *
from dataclasses import dataclass, replace
from ..io import RegionSet
from ..region2vec import Region2VecExModel
from .utils import *
from ..models import Model
from ..search import QdrantBackend
import tensorflow as tf
import matplotlib as plt


@dataclass
class BedMetadataEmbedding:
    """
    Store the information of a bed file, its metadata, and embedding
    """
    file_name: str  # the name of the bed file
    metadata: str  # the metadata of the bed file
    region_set: RegionSet  # the RegionSet that contains intervals in that bed file, not tokenized
    metadata_embedding: np.ndarray  # the embedding vector of the metadata by sentence transformer
    region_set_embedding: np.ndarray  # the embedding vector of region set


@dataclass
class BedMetadataEmbeddingSet:
    """
    Data set to train the feedforward neural network (X: metadata embeddings; Y: region set embedding)
    """
    training: List[BedMetadataEmbedding]
    validating: List[BedMetadataEmbedding]
    testing: List[BedMetadataEmbedding]

    def __len__(self):
        """
        return sum of number of BedMetadataEmbedding in all data set

        :return:
        """
        return len(self.training) + len(self.validating) + len(self.testing)

    @property
    def tolist(self) -> List[BedMetadataEmbedding]:
        """
        put all BedMetadataEmbedding into one list

        :return: a list of all BedMetadataEmbedding in the set
        """
        result = []
        # add elements from each list
        result.extend(self.training)
        result.extend(self.validating)
        result.extend(self.testing)

        return result

    def generate_data(self, set_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        With given dataset name, return a tuple of X and Y

        :param set_name: "training", "validating", or "testing"
        :return:
        """

        if set_name == "training":
            output_list = self.training
        elif set_name == "validating":
            output_list = self.validating
        elif set_name == "testing":
            output_list = self.testing
        else:
            raise Exception("Please make sure the name of dataset is correct")

        X = []
        Y = []
        for dc in output_list:
            # X: metadata embedding
            X.append(dc.metadata_embedding)
            # Y: bed file embedding
            Y.append(dc.region_set_embedding)

        return np.array(X), np.array(Y)

    def to_qd_upload(self) -> Tuple[np.ndarray, List[str]]:
        """
        return a tuple of:
        a np.ndarray of dimension (number of vectors, vector length), and a list of matching file names

        which matches the uploading format of QdrantBackend
        :return:
        """
        vecs = []
        labels = []
        for embeddings in self.tolist:
            vecs.append(embeddings.region_set_embedding)
            labels.append(embeddings.file_name)
        return np.vstack(vecs), labels


def build_BedMetadataSet_from_files(bed_folder: str,
                                    metadata_path: str,
                                    r2v_model: Region2VecExModel,
                                    st_model: SentenceTransformer) -> BedMetadataEmbeddingSet:
    # training_files, validating_files, testing_files = data_file_split(bed_folder)
    """
    With a folder of bed files and a metadata file (first column matches bed file names),
    embed the bed files with a Region2VecExmodel, and embed metadata with Sentence Transformer,
    returns build a BedMetadataEmbeddingSet

    :param bed_folder: foldler of bed files
    :param metadata_path: metadata file
    :param r2v_model: Region2VecExModel
    :param st_model: sentence transformer model
    :return:
    """
    # sorted list of file names in the bed file folder
    file_name_list = os.listdir(bed_folder)
    file_name_list.sort()

    # create a list of BedMetadataEmbedding
    all_bed_metadata_dc = build_BedMetadata_list(bed_folder,
                                                 file_name_list,
                                                 metadata_path,
                                                 r2v_model,
                                                 st_model)
    # split the training, validating, and testing sets
    training, validating, testing = data_split(all_bed_metadata_dc)
    return BedMetadataEmbeddingSet(training, validating, testing)


def build_BedMetadata_list(bed_folder: str,
                           file_name_list: List[str],
                           metadata_path: str,
                           r2v_model: Region2VecExModel,
                           st_model: SentenceTransformer) -> List[BedMetadataEmbedding]:
    """
    With each bed file in the given folder and its matching metadata from the meadata file,
    create a BedMetadataEmbedding with each, and return the list containing all.

    :param bed_folder:
    :param file_name_list:
    :param metadata_path:
    :param r2v_model:
    :param st_model:
    :return:
    """
    output_list = []

    # read the lines from the metadata file
    with open(metadata_path) as m:
        metadata_lines = m.readlines()

    # index to traverse metadata and file list
    # make sure metadata is sorted by the name of interval set
    # this can be done by this command
    # sort -k1 1 metadata_file >  new_metadata_file
    i = 0
    j = 0

    while i < len(metadata_lines):

        # read the line of metadata
        metadata_line = metadata_lines[i]
        # get the name of the interval set
        set_name = metadata_line.split("\t")[0]

        if j < len(file_name_list) and file_name_list[j].startswith(set_name):
            bed_file_name = file_name_list[j]
            bed_file_path = os.path.join(bed_folder, bed_file_name)
            bed_metadata = metadata_line_process(metadata_line)
            region_set = RegionSet(bed_file_path)
            metadata_embedding = st_model.encode(bed_metadata)
            region_set_embedding = np.nanmean(r2v_model.encode(region_set), axis=0)
            bed_metadata_dc = BedMetadataEmbedding(bed_file_name,
                                                   bed_metadata,
                                                   region_set,
                                                   metadata_embedding,
                                                   region_set_embedding)
            output_list.append(bed_metadata_dc)
            j += 1

        # end the loop if all
        if j == len(file_name_list):
            break

        i += 1

    # print a message if not all bed files are matched to metadata rows
    if i < j:
        print("An incomplete list will be returned, some files cannot be matched to any rows by first column")

    return output_list


def update_BedMetadata_list(old_list: List[BedMetadataEmbedding],
                            r2v_model: Region2VecExModel) -> List[BedMetadataEmbedding]:
    """
    With an old list of BedMetadataEmbedding, re-embed the region set with a new region2vec model,
    then return the list of new BedMetadataEmbedding with re-embedded region set vectors.
    :param old_list:
    :param r2v_model:
    :return:
    """
    new_list = []
    for dc in old_list:
        new_dc = replace(dc, region_set_embedding=r2v_model.encode(dc.region_set))
        new_list.append(new_dc)

    return new_list


class TextToBedNN(Model):
    """A Extended Model for TextToBed using a FFNN."""

    def __init__(self,
                 model_nn_path: Union[str, None] = None,
                 model_st_repo: str = DEFAULT_HF_ST_MODEL,
                 **kwargs):
        # super().__init__(model, universe, tokenizer)
        self.built = True
        self._st_model = SentenceTransformer(model_st_repo)
        if model_nn_path is not None:
            # upload local models
            self.from_pretrained(model_nn_path)
        else:
            # create an empty Sequential
            self._nn_model = tf.keras.models.Sequential()
            self.built = False # model is not built
        # initiation

    def __repr__(self):
        print("To be completed")

    def init_from_huggingface(self):
        # to be finished
        print("To be completed")

    def from_pretrained(self, model_nn_path: str):
        """
        load pretrained model from local file

        :param model_nn_path: the local path of saved model
        :return:
        """
        self._nn_model = tf.keras.models.load_model(model_nn_path)

    def train(self, dataset: BedMetadataEmbeddingSet,
              loss_func: str = DEFAULT_LOSS,
              epochs: int = DEFAULT_EPOCHES,
              batch_size: int = DEFAULT_BATCH_SIZE,
              units: int = DEFAULT_UNITS,
              layers: int = DEFAULT_LAYERS,
              plotting: bool = False):
        """
        train the feedforward neural network model with a given BedMetadataEmbeddingSet

        :param dataset: a BedMetadataEmbeddingSet
        :param loss_func: loss function for training
        :param epochs: number of training epochs
        :param batch_size: training batch size
        :param units: number of neurons in a dense layer
        :param layers: number of extra layers
        :param plotting: whether to plot the training and velidation loss by epochs
        :return:
        """

        # get training and validating dataset from BedMetadataEmbeddingSet
        training_X, training_Y = dataset.generate_data("training")
        validating_X, validating_Y = dataset.generate_data("validating")

        # if the model is empty
        if not self.built:
            print("Build new sequential model")

            # dimensions of input and output
            input_dims = training_X[0].shape
            output_dims = training_Y[0].shape

            # the dense layer that connected to input
            self._nn_model.add(tf.keras.layers.Dense(units, input_shape=input_dims, activation='relu'))

            # extra dense layers
            for i in range(layers):
                self._nn_model.add(tf.keras.layers.Dense(units, input_shape=(units,), activation='relu'))

            # output
            self._nn_model.add(tf.keras.layers.Dense(output_dims[0]))

            self._nn_model.compile(optimizer="adam", loss=loss_func)

            # model is no longer empty
            self.built = True

        # early stoppage to prevent overfitting
        early_stoppage = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(epochs * 0.25))

        # record training history
        train_hist = self._nn_model.fit(training_X, training_Y, epochs=epochs, batch_size=batch_size,
                                        validation_data=(validating_X, validating_Y),
                                        callbacks=[early_stoppage])

        # plot the training history: training / validation loss by epoch
        if plotting:
            epoch_range = range(1, len(train_hist.history['loss']) + 1)
            train_loss = train_hist.history['loss']
            valid_loss = train_hist.history['val_loss']
            plt.plot(epoch_range, train_loss, 'r', label='Training loss')
            plt.plot(epoch_range, valid_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()
            plt.show()

    def forward(self, query: str) -> np.ndarray:
        """
        map a query string to an embedding vector of bed file

        :param query: a string
        :return:
        """

        # input string -> embedded by sentence transformer -> FNN -> vector

        # encode the query string to vector by sentence transformer
        query_embedding = self._st_model.encode(query)

        # reshape the encoding vector
        # because the model takes input with shape of (n, <vector dimension>)
        # and the shape of encoding vector is (<vector dimension>, )
        vec_dim = query_embedding.shape[0]

        # map the encoding vector to bedfile embedding vector by feedforward neural network
        output_vec = self._nn_model.predict(query_embedding.reshape((1, vec_dim)))

        # reshape the output from (1, <region2vec embedding dimension>) to (<~ dimension>, 1)
        output_vec_dim = output_vec.shape[1]
        return output_vec.reshape(output_vec_dim, )


class TextToBedNNSearchInterface(object):
    def __init__(self, nn_model: TextToBedNN, region_set_backend: QdrantBackend):
        self.model = nn_model
        self.region_set_backend = region_set_backend

    def nlsearch(self, query: str, k: int = 10) -> List:
        """
        Given an input natural language, suggest region sets

        :param query: searching input string
        :param k: number of results (nearst neighbor in vectors)
        :return: a list of Qdrant Client search results
        """

        # first, get the embedding of the query string
        query_embedding = self.model.forward(query)
        # then, use the query embedding to predict the region set embedding
        # region_set_embedding = self.tum.str_to_region_set(query_embedding)
        # finally, use the region set embedding to search for similar region sets
        return self.region_set_backend.search(query_embedding, k)

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