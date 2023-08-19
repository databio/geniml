from typing import List, Union

from ..io import Region
from .const import DEFAULT_MAX_LENGTH
from .schemas import EncodedRegions
from .utils import wordify_region, unwordify_region


class RegionIDifier:
    """
    Class for converting a region to an integer id. This class
    is intended to be used as a preprocessing step for
    the ATAC Transformer model.
    """

    def __init__(self, vocab_file: str = None, max_length: int = DEFAULT_MAX_LENGTH):
        self._word_to_id = {}
        self._id_to_word = {}
        self.max_length = max_length

        if vocab_file is not None:
            self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file: str):
        """
        Load the vocab file. Read in the file and
        create a mapping from word to id and id to word.
        """
        with open(vocab_file, "r") as f:
            for i, line in enumerate(f.readlines()):
                line = line.strip()
                self._word_to_id[line] = i
                self._id_to_word[i] = line

    def word_to_id(self, word: str) -> int:
        """
        Convert a word to an id.
        """
        _id = self._word_to_id.get(word, None)
        if _id is None:
            _id = self.word_to_id("[UNK]")
        return _id

    def id_to_word(self, id: int) -> str:
        """
        Convert an id to a word.
        """
        return self._id_to_word.get(id, "[UNK]")

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert a list of ids to a list of words.
        """
        return [self.id_to_word(i) for i in ids]

    def convert_ids_to_regions(self, ids: List[int]) -> List[str]:
        """
        Convert a list of ids to a list of regions.
        """
        return [unwordify_region(self.id_to_word(i)) for i in ids]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert a list of tokens to a list of ids.
        """
        return [self.word_to_id(t) for t in tokens]

    def convert_regions_to_ids(self, regions: List[Region]) -> List[int]:
        """
        Convert a list of regions to a list of ids.
        """
        return [self.word_to_id(wordify_region(r)) for r in regions]

    def generate_attention_mask_from_ids(self, ids: List[int]) -> List[int]:
        """
        Generate an attention mask for a list of ids. Will be 1 unless the
        id is the id for the [PAD] token or the [MASK] token.
        """
        return [1 if i != self.word_to_id("[PAD]") else 0 for i in ids]

    def generate_attention_mask_from_tokens(self, tokens: List[str]) -> List[int]:
        """
        Generate an attention mask for a list of tokens.
        """
        return [1 if t != "[PAD]" else 0 for t in tokens]

    def pad_ids(self, ids: List[int], max_length: int = None) -> List[int]:
        """
        Pad a list of ids to a maximum length.
        """
        max_length = max_length or self.max_length
        return ids + [self.word_to_id("[PAD]")] * (max_length - len(ids))

    def pad_tokens(self, regions: List[str], max_length: int = None) -> List[str]:
        """
        Pad a list of regions to a maximum length.
        """
        max_length = max_length or self.max_length
        return regions + ["[PAD]"] * (max_length - len(regions))

    def truncate_ids(self, ids: List[int], max_length: int = None) -> List[int]:
        """
        Truncate a list of ids to a maximum length.
        """
        max_length = (max_length or self.max_length) + 1  # add 1 for [CLS]
        return ids[:max_length]

    def truncate_tokens(self, regions: List[str], max_length: int = None) -> List[str]:
        """
        Truncate a list of regions to a maximum length.
        """
        max_length = (max_length or self.max_length) + 1  # add 1 for [CLS]
        return regions[:max_length]

    def tokenize(
        self,
        regions: Union[List[Region], List[List[Region]]],
        max_length: int = None,
        padding: bool = True,
    ) -> Union[EncodedRegions, List[EncodedRegions]]:
        """
        Encode regions into EncodedRegions objects. Can accept
        either a single region list or multiple lists of regions.

        If multiple are passed, then the regions will be concatenated
        and the attention mask will be generated accordingly. In addition,
        the regions will be padded to the maximum length of all the regions.
        """

        # identify max length
        if max_length is None:
            max_length = self.max_length

        # standardize to list of lists
        if isinstance(regions[0], Region):
            regions = [regions]

        # add [CLS] token to beginning of each region list
        regions = [["[CLS]"] + [wordify_region(r) for r in r_list] for r_list in regions]

        # truncate all to max_length
        regions = [self.truncate_tokens(r, max_length) for r in regions]

        if padding:
            # find max of all lengths
            max_len = max([len(r) for r in regions])

            # pad to max length
            regions = [self.pad_tokens(r, max_len) for r in regions]

        # convert to ids
        ids = [[self.word_to_id(w) for w in r] for r in regions]

        # generate attention mask
        attention_mask = [self.generate_attention_mask_from_ids(id_list) for id_list in ids]

        # generate encoded region objects
        encoded_regions = [
            EncodedRegions(ids=r, attention_mask=a) for r, a in zip(ids, attention_mask)
        ]

        if len(encoded_regions) == 1:
            return encoded_regions[0]
        else:
            return encoded_regions

    def __call__(
        self, regions: Union[List[Region], List[List[Region]]], **kwargs
    ) -> Union[EncodedRegions, List[EncodedRegions]]:
        return self.tokenize(regions, **kwargs)
