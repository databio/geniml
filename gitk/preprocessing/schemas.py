from typing import List
from pydantic import BaseModel


class EncodedRegion(BaseModel):
    """
    A region that has been encoded into a list of ids.
    """

    ids: List[int]
    attention_mask: List[int]
