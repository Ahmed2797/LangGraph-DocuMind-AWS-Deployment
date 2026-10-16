import time
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings
from src.logger import logging
from src.chat_model import get_embeddings
from src.constants import EXPECTED_DIMENSION, EXPECTED_METRIC ,CLOUD, REGION,READY_TIMEOUT_S


@lru_cache(maxsize=1)
def _get_client() -> Pinecone:
    api_key = get_settings().pinecone_api_key
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required to use Pinecone.")
    return Pinecone(api_key=api_key)


def _wait_until_ready(pc: Pinecone, index_name: str) -> None:
    """A new serverless index can't be queried/upserted for a few seconds."""
    deadline = time.time() + READY_TIMEOUT_S
    while time.time() < deadline:
        status = pc.describe_index(index_name).status
        ready = status["ready"] if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            return
        time.sleep(2)
    raise TimeoutError(f"Pinecone index '{index_name}' was not ready after {READY_TIMEOUT_S}s.")


@lru_cache(maxsize=8)
def create_pincone_database(index_name: str):
    """
    Return a connected Pinecone index (created + validated on first use, then
    cached for the life of the process — no extra API calls per request).

    Raises:
        ValueError: missing API key / name, dimension or metric mismatch.
        TimeoutError: a newly created index did not become ready.
    (Exceptions are not cached, so a failed call is retried next time.)
    """
    if not index_name:
        raise ValueError("A Pinecone index name is required.")

    pc = _get_client()

    if index_name not in {idx.name for idx in pc.list_indexes()}:
        logging.info(f"Creating Pinecone index '{index_name}'...")
        try:
            pc.create_index(
                name=index_name,
                dimension=EXPECTED_DIMENSION,
                metric=EXPECTED_METRIC,
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            )
        except Exception as e:                       # another request created it first
            if "already exists" not in str(e).lower():
                raise
        _wait_until_ready(pc, index_name)

    info = pc.describe_index(index_name)
    if info.dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Dimension mismatch for Pinecone index '{index_name}': expected "
            f"{EXPECTED_DIMENSION}, found {info.dimension}. Use a matching index or a new one."
        )
    if str(info.metric).lower() != EXPECTED_METRIC:
        raise ValueError(
            f"Metric mismatch for Pinecone index '{index_name}': expected "
            f"'{EXPECTED_METRIC}', found '{info.metric}'. Retrieval thresholds assume cosine."
        )

    logging.info(f"Connected to Pinecone index '{index_name}'.")
    return pc.Index(index_name)



def create_embedding(text: str) -> list[float]:
    # embed_query takes a string and returns a list of floats
    return get_embeddings().embed_query(text)

