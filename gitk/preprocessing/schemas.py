from typing import List
from pydantic import BaseModel


class EncodedRegions(BaseModel):
    """
    A region that has been encoded into a list of ids.
    """

    ids: List[int]
    attention_mask: List[int]


class TokenMask(BaseModel):
    """
    A token mask. Contains the masked ids and the indices of the masked tokens.
    """

    ids: List[int]
    indices: List[int]
