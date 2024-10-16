import logging
import math
from typing import Dict, List, Tuple, Union

import numpy as np

from ...const import PKG_NAME
from .abstract import EmSearchBackend

_LOGGER = logging.getLogger(PKG_NAME)


def batch_bed_vectors(
    matching_beds: List[Dict], text_results: Union[None, List[int], List[float]] = None
) -> Tuple[np.ndarray, List]:
    """
    Stack the embedding vector of bed files related to a metadata tag together for batch search

    :param matching_beds: result of BED retrieval from Qdrant Client by ids
    :param text_results: keep track of rank or score from metadata embedding search that matches each query bed
    """

    bed_vectors = []
    valid_text_results = []
    for idx, bed in enumerate(matching_beds):
        try:
            bed_vec = bed["vector"]
            bed_vectors.append(bed_vec)
            # only keep metadata embedding score / rank of valid bed vectors
            if text_results is not None:
                valid_text_results.append(text_results[idx])
        except KeyError:
            _LOGGER.warning(f"Retrieved result missing vector: {bed}")
            continue
        except TypeError:
            _LOGGER.warning(
                f"Please check the data loading; retrieved result is not a dictionary: {bed}"
            )
            continue
    return np.array(bed_vectors), valid_text_results


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
        """
        :param metadata_backend: search backend where embedding vectors of metadata tags are stored
        :param bed_backend: search backend where embedding vectors of BED files are stored
        :param metadata_payload_matches: the key in metadata backend payloads to files matching to that metadata tag
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
        """
        :param query: query vector (embedding vector of query term)
        :param limit: number of nearest neighbors to search for query vector
        :param with_payload: whether payload is included in the result
        :param with_vectors: whether the stored vector is included in the result
        :param offset: the offset of the search results
        :param p: weights to the score of metadata search, recommend 0 < p <= 1.0
        :param q: weights to the score of BED search, recommend 0 < q <= 1.0
        :param distance: whether the score is distance or similarity
        :param rank: whether the result is ranked based on rank or score
        :return: the search result(a list of dictionaries,
            each dictionary include: storage id, vector payload (optional), vector (optional))
        """

        # the key for the score in result: distance or score (cosine similarity)
        self.score_key = "distance" if distance else "score"

        # metadata search
        metadata_results = self.metadata_backend.search(
            query,
            limit=int(math.log(limit) * 5) if limit > 10 else 10,
            with_payload=True,
            offset=offset,
        )

        if not isinstance(metadata_results, list):
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
        """
        Search based on maximum rank in results of metadata embedding and results of BED embedding

        :param metadata_results: result of metadata search
        :param limit: see docstring of def search
        :param with_payload:
        :param with_vectors:
        :param offset:
        :return: the search result ranked based on maximum rank
        """

        text_rank = []
        query_beds = []

        query_bed_ids = set()

        for i, result in enumerate(metadata_results):
            # all bed files matching the retrieved metadata tag
            bed_ids = result["payload"][self.metadata_payload_matches]

            unique_bed_ids = [id_ for id_ in bed_ids if id_ not in query_bed_ids]
            query_bed_ids.update(unique_bed_ids)
            matching_beds = self.bed_backend.retrieve_info(unique_bed_ids, with_vectors=True)
            if not isinstance(matching_beds, list):
                matching_beds = [matching_beds]
            for retrieved in matching_beds:
                text_rank.append(i)
                query_beds.append(retrieved)

        bed_vecs, matching_text_rank = batch_bed_vectors(query_beds, text_rank)

        # search request once
        retrieved_batch = self.bed_backend.search(
            bed_vecs,
            limit=limit * 2 if limit < 500 else 500,
            with_payload=with_payload,
            with_vectors=with_vectors,
            offset=offset,
        )

        bed_results = []
        max_rank = []

        for i, retrieved_beds in enumerate(retrieved_batch):
            # j: rank for each bed vector query search
            for j, retrieval in enumerate(retrieved_beds):
                bed_results.append(retrieval)
                # collect maximum rank
                max_rank.append(max(matching_text_rank[i], j))

        return self._top_k(max_rank, bed_results, limit, True)

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
        """
        Search based on weighted score from results of metadata embedding and results of BED embedding

        :param metadata_results: result of metadata search
        :param limit: see docstring of def search
        :param with_payload:
        :param with_vectors:
        :param offset:
        :param p:
        :param q:
        :return: the search result ranked based on weighted similarity scores
        """
        text_scores = []
        query_beds = []

        query_bed_ids = set()

        for result in metadata_results:
            # similarity score between query term and metadat tag
            text_score = (
                1 - result[self.score_key]
                if self.score_key == "distance"
                else result[self.score_key]
            )
            bed_ids = result["payload"][self.metadata_payload_matches]
            unique_bed_ids = [id_ for id_ in bed_ids if id_ not in query_bed_ids]
            query_bed_ids.update(unique_bed_ids)
            matching_beds = self.bed_backend.retrieve_info(unique_bed_ids, with_vectors=True)
            if not isinstance(matching_beds, list):
                matching_beds = [matching_beds]
            for retrieved in matching_beds:
                text_scores.append(text_score)
                query_beds.append(retrieved)

        bed_vecs, matching_text_scores = batch_bed_vectors(query_beds, text_scores)

        retrieved_batch = self.bed_backend.search(
            bed_vecs,
            limit=limit * 2 if limit < 500 else 500,
            with_payload=with_payload,
            with_vectors=with_vectors,
            offset=offset,
        )

        bed_results = []
        overall_scores = []

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
                overall_scores.append((p * matching_text_scores[i] + q * bed_score) / 2)

        return self._top_k(overall_scores, bed_results, limit, False)

    def _top_k(
        self,
        scales: List[Union[int, float]],
        results: List[Dict[str, Union[int, float, Dict[str, str], List[float]]]],
        k: int,
        rank: bool = True,
    ):
        """
        Sort top k result and remove repetition

        :param scales: list of weighted scores or maximum rank
        :param results: retrieval result
        :param k: number of result to return
        :param rank: whether the scale is maximum rank or not
        :return: the top k selected result after rank
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

        top_k_results = list(unique_result.values())[:k]
        return top_k_results
