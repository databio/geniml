from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from .const import *
from fastapi import Depends, Header, Form
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBearer
from typing import Union
import os


# from pephub/pephub/dependencies.py
def get_qdrant_enabled() -> bool:
    """
    Check if qdrant is enabled
    """
    return parse_boolean_env_var(os.environ.get("QDRANT_ENABLED", "false"))


def get_qdrant(
    qdrant_enabled: bool = Depends(get_qdrant_enabled),
) -> Union[QdrantClient, None]:
    """
    Return connection to qdrant client
    """
    # return None if qdrant is not enabled
    if not qdrant_enabled:
        try:
            yield None
        finally:
            pass
    # else try to connect, test connectiona and return client if connection is successful.
    qdrant = QdrantClient(
        url=os.environ.get("QDRANT_HOST", DEFAULT_QDRANT_HOST),
        port=os.environ.get("QDRANT_PORT", DEFAULT_QDRANT_PORT),
        api_key=os.environ.get("QDRANT_API_KEY", None),
    )
    try:
        # test the connection first
        qdrant.list_full_snapshots()
        yield qdrant
    except ResponseHandlingException as e:
        print(f"Error getting qdrant client: {e}")
        yield None
    finally:
        # no need to close the connection
        pass


def parse_boolean_env_var(env_var: str) -> bool:
    """
    Helper function to parse a boolean environment variable
    """
    return env_var.lower() in ["true", "1", "t", "y", "yes"]