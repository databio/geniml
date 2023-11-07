from collections import Counter

import scanpy as sc
from qdrant_client import QdrantClient
from rich.progress import track


class AnnotationServer(QdrantClient):
    def __init__(
        self,
        location: str = None,
        url: str = None,
        port: int = None,
        api_key: str = None,
        collection_name: str = None,
        timeout: float = 10,
    ):
        """
        A class for querying a Qdrant server for cell type predictions. This class requires that you have
        a Qdrant server running with a collection of cell type embeddings. You can create this collection using
        the `geniml/examples/scembed/load_qdrant.ipynb` script.

        :param str collection_name: The name of the collection to query.
        :param str location: The location of the Qdrant server. This should be in the format `host:port`.
        :param str url: The URL of the Qdrant server.
        :param int port: The port of the Qdrant server.
        :param str api_key: The API key for the Qdrant server.
        :param float timeout: The timeout for the Qdrant server.
        """
        super().__init__(location, url=url, port=port, api_key=api_key, timeout=timeout)
        self.collection_name = collection_name


class Annotator:
    def __init__(
        self,
        collection_name: str = None,
        location: str = None,
        url: str = None,
        port: int = None,
        timeout: float = 10,
    ):
        """
        A class for annotating single cell data with cell type predictions. This class requires that you have
        a Qdrant server running with a collection of cell type embeddings. You can create this collection using
        the `geniml/examples/scembed/load_qdrant.ipynb` script.

        :param str collection_name: The name of the collection to query.
        :param str location: The location of the Qdrant server. This should be in the format `host:port`.
        :param str url: The URL of the Qdrant server.
        :param int port: The port of the Qdrant server.
        :param float timeout: The timeout for the Qdrant server.

        """
        self.collection_name = collection_name
        self.url = url
        self.port = port
        self.timeout = timeout
        self._annotation_server = AnnotationServer(
            location=location,
            url=self.url,
            port=self.port,
            collection_name=self.collection_name,
            timeout=self.timeout,
        )

    def annotate(
        self,
        adata: sc.AnnData,
        embedding_key: str = "embedding",
        key_added: str = "pred_celltype",
        cluter_key: str = "leiden",
        knn: int = 3,
        score_threshold: float = 0.5,
    ):
        """
        Annotate a sc.AnnData object with cell type predictions for each cluster. This functions requires
        that you have 1) clustered your data and 2) embedded your data using some pretrained model (databio/multiome).

        You can cluster your data using the `scanpy`

        It is *imperative* that the model used to embed your single cells is the same model used to produce the
        embeddings in the database. Otherwise, the predictions will not be accurate; in fact they will be meaningless.

        :param sc.AnnData adata: The annotated data.
        :param str embedding_key: The key in `adata.obsm` where the embeddings are stored.
        :param str key_added: The key in `adata.obs` where the cell type predictions will be stored.
        :param str cluter_key: The key in `adata.obs` where the cluster labels are stored.
        :param int knn: The number of nearest neighbors to use when querying the database.
        :param float score_threshold: The score threshold to use when querying the database.
        """

        _temp_key = "putative_cell_type"

        # check that the embedding key exists
        if embedding_key not in adata.obsm.keys():
            raise ValueError(
                f"Embedding key '{embedding_key}' not found in adata.obsm. Please embed your data first."
            )
        if cluter_key not in adata.obs.keys():
            raise ValueError(
                f"Cluster key '{cluter_key}' not found in adata.obs. Please cluster your data first."
            )

        # init list
        scembed_cell_type_preds = []

        # use qdrant and a simple KNN approach to attach cell types to the embeddings
        for embedding in track(adata.obsm["embedding"], total=len(adata.obsm["embedding"])):
            results = self._annotation_server.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=knn,
                score_threshold=score_threshold,
            )

            result_dicts = [result.dict() for result in results]

            # count "cell_type" in all dicts
            c = Counter([result["payload"]["cell_type"] for result in result_dicts])

            # simply get the name of the top most common and thats it
            try:
                cell_type = c.most_common(1)[0][0]
            except IndexError:
                cell_type = "Unknown"

            scembed_cell_type_preds.append(cell_type)

        # add to adata, these are just putative cell types
        # we will take a consensus vote later using clusters
        adata.obs[_temp_key] = scembed_cell_type_preds

        # now take a consensus vote for each cluster
        cluster_celltypes = {}
        for cluster in adata.obs["leiden"].unique():
            cluster_celltypes[cluster] = Counter(
                adata.obs[adata.obs["leiden"] == cluster][_temp_key]
            ).most_common(1)[0][0]

        # map the cluster_to_cell_type dictionary to the leiden column
        adata.obs[key_added] = adata.obs[cluter_key].map(cluster_celltypes)
