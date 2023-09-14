from typing import List, Union

from gtokenizers import InMemTokenizer, Region as GRegion

from ..io.io import Region, RegionSet
from .main import Tokenizer


class GTokenizer(Tokenizer):
    """
    A rust-backed genomic region tokenizer that very quickly
    tokenizes a list of regions into known regions from a
    given universe or vocabulary
    """

    def __init__(self, path_to_vocab: str = None):
        """
        Create a new GTokenizer object.

        :param path_to_vocab: path to the vocabulary/universe
        """
        if path_to_vocab is not None:
            self._tokenizer = self._init_tokenizer(path_to_vocab)
        else:
            self._tokenizer = None

    def _init_tokenizer(self, path_to_vocab: str) -> InMemTokenizer:
        """
        Initialize the tokenizer object

        :param path_to_vocab: path to the vocabulary/universe
        """
        return InMemTokenizer(path_to_vocab)

    def add_vocab(self, path_to_vocab: str):
        self._tokenizer = self._init_tokenizer(path_to_vocab)

    def tokenize(self, regions: Union[Region, List[Region], RegionSet, str]) -> List[Region]:
        """
        Tokenize regions into the vocab of the model

        :param regions: regions to tokenize
        :return: tokenized regions
        """
        if isinstance(regions, str):
            regions = RegionSet(regions)
        elif isinstance(regions, Region):
            regions = [regions]
        elif isinstance(regions, RegionSet):
            pass
        elif isinstance(regions, list) and isinstance(regions[0], Region):
            pass
        else:
            raise ValueError("regions must be of type str, Region, RegionSet, or List[Region]")

        results = []
        for region in regions:
            _r = GRegion(region.chr, region.start, region.end)
            result = self._tokenizer.tokenize_region(_r)

            # we iterate this way, since I havnt figured out how to
            # implement __iter__ in the rust code
            # https://pyo3.rs/v0.19.2/class/protocols.html?highlight=__iter__#iterable-objects
            total_regions = len(result)
            for i in range(total_regions):
                r = result.tokens[i]
                results.append(Region(r.chr, r.start, r.end))
        return results
