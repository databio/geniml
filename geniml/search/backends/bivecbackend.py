import logging
import math
from typing import Dict, List, Tuple, Union

import numpy as np

from ...const import PKG_NAME
from .abstract import EmSearchBackend

_LOGGER = logging.getLogger(PKG_NAME)


def batch_for_request(
    ids: Union[List[str], List[int]], ranks_scores: List[float], batch_size: int = 100
) -> List[Tuple[List, List]]:  # used ChatGPT
    """Create batches of BED ids and scores/ranks for requests.

    Args:
        ids: collected ids of BED files matching retrieved metadata tags
        ranks_scores: keep track of ranks or scores from metadata tag embedding vector search
        batch_size: size of batch, > 100 may crash qdrant server

    Returns:
        Batched BED ids and matching text-search scores/ranks as list of tuples.
    """
    # Check if the lists are the same length
    if len(ids) != len(ranks_scores):
        raise ValueError("The lists must have the same length.")

    # Create batches
    batches = []
    for i in range(0, len(ids), batch_size):
        batch1 = ids[i : i + batch_size]
        batch2 = ranks_scores[i : i + batch_size]
        batches.append((batch1, batch2))

    return batches


class BiVectorBackend:
    """
    Search backend that connects the embeddings of metadata tags and bed files
    """

    def __init__(
        self,
        metadata_backend: EmSearchBackend,
        bed_backend: EmSearchBackend,
        metadata_payload_matches: str = "matched_files",
    ):
        """Initialize the BiVectorBackend.

        Args:
            metadata_backend: search backend where embedding vectors of metadata tags are stored
            bed_backend: search backend where embedding vectors of BED files are stored
            metadata_payload_matches: the key in metadata backend payloads to files matching to that metadata tag
        """
        self.metadata_backend = metadata_backend
        self.bed_backend = bed_backend
        self.metadata_payload_matches = metadata_payload_matches

    def search(
        self,
        query: np.ndarray,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
        p: float = 1.0,
        q: float = 1.0,
        distance: bool = False,
        rank: bool = False,
    ) -> List[Dict[str, Union[int, float, Dict[str, str], List[float]]]]:
        """Search for nearest neighbors in both metadata and BED embeddings.

        Args:
            query: query vector (embedding vector of query term)
            limit: number of nearest neighbors to search for query vector
            with_payload: whether payload is included in the result
            with_vectors: whether the stored vector is included in the result
            offset: the offset of the search results
            p: weights to the score of metadata search, recommend 0 < p <= 1.0
            q: weights to the score of BED search, recommend 0 < q <= 1.0
            distance: whether the score is distance or similarity
            rank: whether the result is ranked based on rank or score

        Returns:
            The search result as a list of dictionaries, each containing storage id,
            vector payload (optional), and vector (optional).
        """

        # the key for the score in result: distance or score (cosine similarity)
        self.score_key = "distance" if distance else "score"

        # metadata search
        metadata_results = self.metadata_backend.search(
            query,
            limit=int(math.log(limit) * 5) if limit > 10 else 5,
            with_payload=True,
            offset=0,
        )

        if isinstance(metadata_results, dict):
            metadata_results = [metadata_results]

        if rank:
            return self._rank_search(metadata_results, limit, with_payload, with_vectors, offset)
        else:
            return self._score_search(
                metadata_results, limit, with_payload, with_vectors, offset, p, q
            )

    def _rank_search(
        self,
        metadata_results: List[Dict],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
    ) -> List[Dict[str, Union[int, float, Dict[str, str], List[float]]]]:
        """Search based on maximum rank in results of metadata and BED embeddings.

        Args:
            metadata_results: result of metadata search
            limit: see docstring of search()
            with_payload: whether payload is included in the result
            with_vectors: whether the stored vector is included in the result
            offset: the offset of the search results

        Returns:
            The search result ranked based on maximum rank.
        """

        text_rank = []
        ids_to_retrieve = []

        query_bed_ids = set()

        for i, result in enumerate(metadata_results):
            # all bed files matching the retrieved metadata tag
            bed_ids = result["payload"][self.metadata_payload_matches]

            unique_bed_ids = [id_ for id_ in bed_ids if id_ not in query_bed_ids]
            query_bed_ids.update(unique_bed_ids)

            for id_ in unique_bed_ids:
                text_rank.append(i)
                ids_to_retrieve.append(id_)
        bed_results = []
        max_rank = []
        request_batches = batch_for_request(ids_to_retrieve, text_rank, 100)

        for ids, ranks in request_batches:
            query_beds = self.bed_backend.retrieve_info(ids, with_vectors=True)
            if isinstance(query_beds, dict):
                query_beds = [query_beds]

            bed_vecs = [b["vector"] for b in query_beds]

            # search request once
            retrieved_batch = self.bed_backend.search(
                np.array(bed_vecs),
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                offset=0,
            )

            for i, retrieved_beds in enumerate(retrieved_batch):
                # j: rank for each bed vector query search
                for j, retrieval in enumerate(retrieved_beds):
                    bed_results.append(retrieval)
                    # collect maximum rank
                    max_rank.append(max(ranks[i], j))

        return self._top_k(max_rank, bed_results, limit, offset=offset, rank=True)

    def _score_search(
        self,
        metadata_results: List[Dict],
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = True,
        offset: int = 0,
        p: float = 1.0,
        q: float = 1.0,
    ) -> List[Dict[str, Union[int, float, Dict[str, str], List[float]]]]:
        """Search based on weighted score from metadata and BED embeddings.

        Args:
            metadata_results: result of metadata search
            limit: see docstring of search()
            with_payload: whether payload is included in the result
            with_vectors: whether the stored vector is included in the result
            offset: the offset of the search results
            p: weights to the score of metadata search
            q: weights to the score of BED search

        Returns:
            The search result ranked based on weighted similarity scores.
        """
        text_scores = []
        ids_to_retrieve = []

        query_bed_ids = set()
        for i, result in enumerate(metadata_results):
            # all bed files matching the retrieved metadata tag
            text_score = (
                1 - result[self.score_key]
                if self.score_key == "distance"
                else result[self.score_key]
            )
            bed_ids = result["payload"][self.metadata_payload_matches]

            unique_bed_ids = [id_ for id_ in bed_ids if id_ not in query_bed_ids]
            query_bed_ids.update(unique_bed_ids)

            for id_ in unique_bed_ids:
                text_scores.append(text_score)
                ids_to_retrieve.append(id_)

        bed_results = []
        overall_scores = []

        request_batches = batch_for_request(ids_to_retrieve, text_scores, 100)

        for ids, scores in request_batches:
            query_beds = self.bed_backend.retrieve_info(ids, with_vectors=True)
            if isinstance(query_beds, dict):
                query_beds = [query_beds]

            bed_vecs = [b["vector"] for b in query_beds]

            retrieved_batch = self.bed_backend.search(
                np.array(bed_vecs),
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                offset=0,
            )

            for i, retrieved_beds in enumerate(retrieved_batch):
                # j: rank for each bed vector query search
                for retrieval in retrieved_beds:
                    # calculate weighted score
                    bed_score = (
                        1 - retrieval[self.score_key]
                        if self.score_key == "distance"
                        else retrieval[self.score_key]
                    )
                    bed_results.append(retrieval)
                    overall_scores.append((p * scores[i] + q * bed_score) / 2)

        return self._top_k(overall_scores, bed_results, limit=limit, offset=offset, rank=False)

    def _top_k(
        self,
        scales: List[Union[int, float]],
        results: List[Dict[str, Union[int, float, Dict[str, str], List[float]]]],
        limit: int = 10,
        offset: int = 0,
        rank: bool = True,
    ):
        """Sort top k results and remove duplicates.

        Args:
            scales: list of weighted scores or maximum rank
            results: retrieval results
            limit: number of results to return
            offset: the offset of the search results
            rank: whether the scale is maximum rank or not

        Returns:
            The top k selected results after ranking.
        """
        paired_score_results = list(zip(scales, results))

        # sort result
        if not rank:
            paired_score_results.sort(reverse=True, key=lambda x: x[0])
        else:
            paired_score_results.sort(key=lambda x: x[0])

        unique_result = {}
        for scale, result in paired_score_results:
            store_id = result["id"]
            # filter out overlap
            if store_id not in unique_result:
                # add rank or score into the result
                if not rank:
                    if self.score_key == "distance":
                        del result[self.score_key]
                    result["score"] = scale
                else:
                    try:
                        del result["score"]
                    except KeyError:
                        del result["distance"]

                    result["max_rank"] = scale
                unique_result[store_id] = result

        top_k_results = list(unique_result.values())[offset : limit + offset]
        return top_k_results
