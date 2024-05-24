import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
from gensim.models import Word2Vec


def genome_distance(u: Tuple[int, int], v: Tuple[int, int]) -> float:
    """Computes the genome distance between two regions.

    Assumes that the two regions, u and v, are on the same chromosome.

    Args:
        u (tuple[int, int]): A region denoted by its start and end positions.
        v (tuple[int, int]): A region denoted by its start and end positions.

    Returns:
        float: The genome distance between the two regions.
    """
    return float(u[1] < v[1]) * max(v[0] - u[1] + 1, 0) + float(u[1] >= v[1]) * max(
        u[0] - v[1] + 1, 0
    )


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the cosine distance between two embedding vectors.

    Args:
        x (np.ndarray): An embedding vector.
        y (np.ndarray): An embedding vector.

    Returns:
        float: The cosine distance between two embedding vectors.
    """
    return (1 - ((x / np.linalg.norm(x)) * (y / np.linalg.norm(y))).sum()) / 2


class BaseEmbeddings:
    """Wraps embeddings and the corresponding regions in one object.

    Attributes:
        embeddings (np.ndarray): Region embedding vectors.
        vocab (list[str]): A list of regions in the format of chr:start-end.
    """

    def __init__(self, embeddings, vocab):
        self.embeddings = embeddings
        self.vocab = vocab


def get_bin_embeddings(universe_file: str, tokenized_files: list[str]) -> BaseEmbeddings:
    """Gets a BaseEmbeddings object for binary embeddings.

    Args:
        universe_file (str): The path to a universe file.
        tokenized_files (list[str]): A list of tokoenized BED files (in full
            paths).

    Returns:
        BaseEmbeddings: A BaseEmbeddings object for binary embeddings.
    """
    vocab = []
    with open(universe_file, "r") as f:
        for line in f:
            elements = line.strip().split("\t")
            region = f"{elements[0]}:{elements[1]}-{elements[2]}"
            vocab.append(region)
    vocab_dict = {v: i for i, v in enumerate(vocab)}
    print("vocab size is", len(vocab))
    bin_embeds = np.zeros((len(vocab), len(tokenized_files)))
    for i, token_file in enumerate(tokenized_files):
        with open(token_file, "r") as f:
            for line in f:
                elements = line.strip().split("\t")
                region = f"{elements[0]}:{elements[1]}-{elements[2]}"
                if region in vocab_dict:
                    bin_embeds[vocab_dict[region]][i] = 1
    bin_embed_obj = BaseEmbeddings(bin_embeds, vocab)
    return bin_embed_obj


def get_pca_embeddings(
    bin_embed_obj: BaseEmbeddings, dim: int, kwargs: Dict[str, Union[int, float]] = {}
) -> BaseEmbeddings:
    """Gets PCA embeddings from binary embeddings.

    Args:
        bin_embed_obj (BaseEmbeddings): A BaseEmbeddings object for binary embeddings.
        dim (int): Number of dimensions for PCA embeddings.
        kwargs (dict[str, Union[int, float]], optional): Parameters passed to
            PCA. Defaults to {}.

    Returns:
        BaseEmbeddings: A BaseEmbeddings object for PCA embeddings.
    """
    from sklearn.decomposition import PCA

    embeds = PCA(n_components=dim, **kwargs).fit_transform(bin_embed_obj.embeddings)
    pca_embed_obj = BaseEmbeddings(embeds, bin_embed_obj.vocab)
    return pca_embed_obj


def get_umap_embeddings(
    bin_embed_obj: BaseEmbeddings, dim: int, kwargs: Dict[str, Union[int, float]] = {}
) -> BaseEmbeddings:
    """Gets UMAP embeddings from binary embeddings.

    Args:
        bin_embed_obj (BaseEmbeddings): A BaseEmbeddings object for binary embeddings.
        dim (int): Number of dimensions for UMAP embeddings.
        kwargs (dict[str, Union[int, float]], optional): Parameters passed to
            UMAP. Defaults to {}.

    Returns:
        BaseEmbeddings: A BaseEmbeddings object for UMAP embeddings.
    """
    import umap

    embeds = umap.UMAP(n_components=dim, **kwargs).fit_transform(bin_embed_obj.embeddings)
    umap_embed_obj = BaseEmbeddings(embeds, bin_embed_obj.vocab)
    return umap_embed_obj


def save_base_embeddings(base_embed_obj: BaseEmbeddings, file_name: str) -> None:
    """Saves the BaseEmbeddings object to disk.

    Args:
        base_embed_obj (BaseEmbeddings): A BaseEmbeddings object.
        file_name (str): Save the BaseEmbeddings object to file_name.
    """
    with open(file_name, "wb") as f:
        pickle.dump(base_embed_obj, f)


def load_base_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """Loads a BaseEmbeddings object.

    Args:
        path (str): The path to a BaseEmbeddings object.

    Returns:
        tuple[np.ndarray, list[str]]: Embedding vectors and the corresponding
            region list.
    """
    with open(path, "rb") as f:
        base_embed_obj = pickle.load(f)
    return base_embed_obj.embeddings, base_embed_obj.vocab


def load_genomic_embeddings(
    model_path: str, embed_type: str = "region2vec"
) -> Tuple[np.ndarray, List[str]]:
    """Loads genomic region embeddings based on the type.

    Args:
        model_path (str): The path to a saved model.
        embed_type (str, optional): The model type. Defaults to "region2vec".
            Can be "region2vec" or "base".

    Returns:
        tuple[np.ndarray, list[str]]: Embedding vectors and the corresponding
            region list.
    """
    if embed_type == "region2vec":
        model = Word2Vec.load(model_path)
        regions_r2v = model.wv.index_to_key
        embed_rep = model.wv.vectors
        return embed_rep, regions_r2v
    elif embed_type == "base":
        embed_rep, regions_r2v = load_base_embeddings(model_path)
        return embed_rep, regions_r2v


def sort_key(x: str) -> Tuple[int, int]:
    """Extracts chromosome in number and the start position of a region.

    Args:
        x (str): A region in the chr:start-end position.

    Returns:
        tuple[int, int]: Chromosome in number and the start position.
    """
    elements = x.split(":")
    chr_idx = elements[0][3:]
    try:
        idx = int(chr_idx)
    except ValueError:
        idx = 23
        for c in chr_idx:
            idx += ord(c)
    start = int(elements[1].split("-")[0].strip())
    return idx, start


def get_vocab(model_path: str, type: str = "base", ordered: bool = True) -> List[str]:
    """Gets vocab from a model.

    Args:
        model_path (str): The path to a saved model.
        type (str, optional): The embedding type. Defaults to "base".
        ordered (bool, optional): Choose whether to sort the regions. Defaults
            to True.

    Returns:
        list[str]: A list of regions.
    """

    if type == "region2vec":
        model = Word2Vec.load(model_path)
        regions_r2v = model.wv.index_to_key
    elif type == "base":
        _, regions_r2v = load_base_embeddings(model_path)
    if ordered:
        regions_r2v = sorted(regions_r2v, key=sort_key)
    return regions_r2v


def write_vocab(vocab: List[str], file_name: str) -> None:
    """Writes a list of regions to a file.

    Args:
        vocab (list[str]): A list of regions in the format of chr:start-end.
        file_name (str): Saves vocab as file_name.
    """
    with open(file_name, "w") as f:
        for v in vocab:
            elements = v.split(":")
            chr = elements[0].strip()
            s, e = elements[1].split("-")
            s = s.strip()
            e = e.strip()
            f.write(f"{chr}\t{s}\t{e}\n")
