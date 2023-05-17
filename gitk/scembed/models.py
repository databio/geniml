from typing import List, Optional
from pydantic import BaseModel


class ModelParameter(BaseModel):
    embedding_dim: Optional[int]
    description: Optional[str]


class Maintainer(BaseModel):
    name: Optional[str]
    email: Optional[str]


class ModelCard(BaseModel):
    path_to_weights: str
    path_to_universe: str
    name: str
    reference: str
    description: Optional[str]
    tags: Optional[List[str]]
    datasets: Optional[List[str]]
    model_parameters: Optional[List[ModelParameter]]
    model_architecture: Optional[str]
    maintainers: Optional[List[Maintainer]]


class Universe(BaseModel):
    reference: str
    regions: List[str]
    description: Optional[str]
