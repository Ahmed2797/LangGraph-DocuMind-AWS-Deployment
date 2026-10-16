import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from src.state import State

MAX_TEXT_CHARS = 500   # questions can contain personal data -> never log them in full


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _first(d: dict, *keys):
    """First non-None value among several possible key names."""
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def _clip(value, limit: int = MAX_TEXT_CHARS) -> str:
    return str(value or "")[:limit]


def _count(value) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


class ProductionLogger:
    """
    Production logger for monitoring RAG pipeline queries,
    evaluation scores, errors, and performance metrics.

    One JSON object per line -> easy to ship to Loki / ELK / BigQuery.
    """

    def __init__(
        self,
        log_file: str = "logs/rag_production.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.logger = logging.getLogger("rag_production")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            # Rotates at 10 MB, keeps 5 old files.
            # (With several uvicorn workers, send logs to stdout / a collector instead:
            #  rotating one file from many processes is not safe.)
            handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    def log_query(self, state: State, metrics: dict):
        """
        Log a completed or failed RAG query.

        Args:
            state: Final LangGraph state (a dict).
            metrics: Runtime metrics. Optional keys: status, error, latency_ms,
                     tokens_used, query_cost, prompt_tokens, completion_tokens,
                     project_id, thread_id.
        """
        state = _as_dict(state)
        metrics = _as_dict(metrics)

        evaluation = _as_dict(state.get("evaluation"))
        hallucination = _as_dict(evaluation.get("hallucination"))
        groundedness = _as_dict(evaluation.get("groundedness"))
        sycophancy = _as_dict(evaluation.get("sycophancy"))

        media = state.get("media") or []
        media_types = sorted({m.get("type") for m in media if isinstance(m, dict) and m.get("type")})

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": metrics.get("status", "success"),
            "error": metrics.get("error") or None,
            "project_id": metrics.get("project_id"),
            "thread_id": metrics.get("thread_id"),

            # Query
            "question": _clip(state.get("question")),
            "standalone_question": _clip(state.get("standalone_question")),
            "retrieval_query": _clip(state.get("retrieval_query")),
            "had_image": bool(state.get("image_description")),

            # Routing / cache
            "route": state.get("route"),                     # rag | tool | direct
            "is_cached": state.get("cache_hit", False),
            "cache_similarity": state.get("cache_similarity"),
            "cache_saved": state.get("cache_saved"),

            # Retrieval
            "docs_retrieved": _count(state.get("docs")),
            "relevant_docs": _count(state.get("relevant_docs")),
            "retrieval_confidence": state.get("retrieval_confidence"),
            "query_tries": state.get("query_tries"),
            "rewrite_tries": state.get("rewrite_tries"),
            "revise_tries": state.get("retries"),

            # Answer
            "answer_length": len(str(state.get("answer") or "")),
            "confidence_score": state.get("confidence_score"),
            "sources_count": _count(state.get("sources")),
            "media_count": len(media) if isinstance(media, (list, tuple)) else 0,
            "media_types": media_types,

            # RAG verification
            "issup": state.get("issup"),
            "isuse": state.get("isuse"),

            # Evaluation (accepts both key spellings)
            "hallucination_score": _first(hallucination, "score", "hallucination_score"),
            "groundedness_score": _first(groundedness, "score", "groundedness_score"),
            "sycophancy_score": _first(sycophancy, "score", "sycophancy_score"),
            "overall_quality": evaluation.get("overall_quality"),
            "is_reliable": evaluation.get("is_reliable"),

            # Performance
            "latency_ms": metrics.get("latency_ms"),
            "tokens_used": metrics.get("tokens_used"),
            "query_cost": metrics.get("query_cost"),
            "completion_tokens": _first(metrics, "completion_tokens", "Completion tokens"),
            "prompt_tokens": _first(metrics, "prompt_tokens", "Prompt tokens"),
        }

        self.logger.info(json.dumps(log_entry, ensure_ascii=False, default=str))