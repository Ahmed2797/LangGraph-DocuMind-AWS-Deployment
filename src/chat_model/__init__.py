from functools import lru_cache
from typing import Optional
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import get_settings

REQUEST_TIMEOUT_S = 60
MAX_RETRIES = 3

# Reasoning models reject a custom temperature.
_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _api_key() -> str:
    """Single source of truth for the key (same settings object as the rest of the app)."""
    key = get_settings().openai_api_key
    if not key:
        raise ValueError("OPENAI_API_KEY is required to use OpenAI.")
    return key

def get_openai_client():
    """
    Create an OpenAI client using OPENAI_API_KEY
    from environment variables.
    """

    api_key = _api_key()

    if not api_key:
        return None

    return OpenAI(
        api_key=_api_key()
    )


@lru_cache(maxsize=4)
def get_llm(model: Optional[str] = None) -> ChatOpenAI:
    """
    OpenAI chat model (cached per model name).

    get_llm()                     -> model from settings (main answer model)
    get_llm("gpt-4o-mini")        -> e.g. a cheaper model for the grader nodes

    Must support vision if you use image queries (gpt-4o / gpt-4o-mini do).
    """
    name = model or get_settings().openai_model
    kwargs = {
        "model": name,
        "api_key": _api_key(),
        "timeout": REQUEST_TIMEOUT_S,
        "max_retries": MAX_RETRIES,
        "temperature": 0
    }
    if not name.lower().startswith(_NO_TEMPERATURE_PREFIXES):
        kwargs["temperature"] = 0          # deterministic answers
    return ChatOpenAI(**kwargs,stream_usage=True)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """text-embedding-3-small (1536 dimensions), created once and reused."""
    return OpenAIEmbeddings(
        model=get_settings().embedding_model,
        api_key=_api_key(),
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )
