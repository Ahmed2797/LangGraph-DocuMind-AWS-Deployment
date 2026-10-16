from typing import List, TypedDict, Literal, Dict, Optional, Any
from langchain_core.documents import Document

from pydantic import BaseModel, Field

# ============================================================
# Type Aliases
# ============================================================

SUPPORT_STATUS = Literal[
    "fully_supported",
    "partially_supported",
    "no_support",
]

USEFULNESS_STATUS = Literal[
    "useful",
    "not_useful",
]

# --------------------------
# Evaluation
# --------------------------

class EvaluationOutput(BaseModel):
    hallucination_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Hallucination score. 0 means no hallucination, 1 means severe hallucination."
    )

    hallucinated_claims: List[str] = Field(
        default_factory=list
    )

    supported_claims: List[str] = Field(
        default_factory=list
    )

    groundedness_score: float = Field(
        ge=0.0,
        le=1.0
    )

    evidence_alignment: float = Field(
        ge=0.0,
        le=1.0
    )

    completeness: float = Field(
        ge=0.0,
        le=1.0
    )

    accuracy: float = Field(
        ge=0.0,
        le=1.0
    )

    relevance: float = Field(
        ge=0.0,
        le=1.0
    )

    sycophancy_detected: bool

    sycophancy_score: float = Field(
        ge=0.0,
        le=1.0
    )

    user_assumptions: List[str] = Field(
        default_factory=list
    )

    contradictions: List[str] = Field(
        default_factory=list
    )

    reasoning: str

class EvaluationResult(TypedDict):
    """
    Final evaluation result produced by the evaluation engine.
    """

    hallucination: Dict
    groundedness: Dict
    sycophancy: Dict

    overall_quality: float | None
    is_reliable: bool
    evaluation_status: Literal["success", "failed"]


# ============================================================
# Graph State
# ============================================================
"""
State is a TypedDict:
"""

class State(TypedDict, total=False):
    # ---- you already have these ----
    question: str
    route: str
    docs: list
    retrieval_scores: List[float]
    relevant_docs: list
    answer: str
    context: str
    sources: List[str]
    citations: List[dict]
    confidence_score: float
    # issup: Optional[str]
    issup: SUPPORT_STATUS
    evidence: List[str]
    # isuse: Optional[str]
    isuse: USEFULNESS_STATUS
    use_reason: str
    retries: int
    rewrite_tries: int
    retrieval_query: str
    optimized_query: str
    keywords: List[str]
    entities: List[str]
    evaluation: EvaluationResult ## Optional[dict]
    cache_hit: bool
    cache_similarity: float
    cache_saved: bool

    # ---- NEW ----
    standalone_question: str          # history/image resolved question
    query_image: Optional[str]        # URL | data-URL | raw base64 (cleared after use)
    image_description: str
    history: List[dict]               # [{"user": ..., "assistant": ...}] for follow-ups
    query_tries: int                  # query_optimizer loop counter
    hybrid_scores: List[float]
    retrieval_confidence: float
    score_type: str                   # "distance" (FAISS/Chroma) | "similarity" (Pinecone)
    media: List[dict]                 # [{"type": image|chart|table, "url"|"path"|"content", "caption", "source", "page"}]
    web_results: str
    skip_cache: bool                  # input: True = ignore cache this turn
    cache_bypass: bool                # internal copy of skip_cache

# 01. Define the output structure using Pydantic[cite: 1, 4]
# ========================================================
# OptimizedQuery Decision
# ========================================================
class OptimizedQueryOutput(BaseModel):
    optimized_query: str = Field(description="Concise retrieval-friendly query")
    keywords: List[str] = Field(description="Important retrieval keywords")
    entities: List[str] = Field(description="Entities explicitly mentioned in the question")


# ============================================================
# 1. Retrieval Decision
# ============================================================

class RetrieveDecision(BaseModel):
    """
    Determines whether document retrieval is required.
    """

    should_retrieve: bool = Field(
        ...,
        description=(
            "True if external/internal documents are required "
            "to answer the question reliably, otherwise False."
        ),
    )


class QueryRoute(BaseModel):
    """
    Determines how the user question should be answered.
    """

    route: Literal["rag", "tool", "direct"] = Field(
        ...,
        description=(
            "Select the best route for answering the user's question. "
            "'rag' means the question requires information from the "
            "application's internal documents or knowledge base. "
            "'tool' means the question requires current, real-time, "
            "external, web-based, or API information. "
            "'direct' means the question can be answered directly by "
            "the LLM without internal document retrieval or external tools."
        ),
    )


# ============================================================
# 2. Document Relevance Decision
# ============================================================

class RelevanceDecision(BaseModel):
    """
    Determines whether a retrieved document is relevant
    to the user's question.
    """

    is_relevant: bool = Field(
        ...,
        description=(
            "True ONLY if the document contains information "
            "that can directly help answer the question."
        ),
    )


# ============================================================
# 3. Answer Support / IsSUP Decision
# ============================================================

class IsSUPDecision(BaseModel):
    """
    Verifies whether the generated answer is supported
    by the retrieved context.
    """

    issup: SUPPORT_STATUS = Field(
        ...,
        description=(
            "Degree to which the answer is supported by "
            "the provided context."
        ),
    )

    evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Short evidence quotes from the context "
            "supporting the generated answer."
        ),
    )


# ============================================================
# 4. Answer Usefulness / IsUSE Decision
# ============================================================

class IsUSEDecision(BaseModel):
    """
    Determines whether the generated answer is useful
    for the user's original question.
    """

    isuse: USEFULNESS_STATUS = Field(
        ...,
        description="Whether the generated answer is useful to the user.",
    )

    use_reason: str = Field(
        ...,
        description="Short one-line explanation of the usefulness decision.",
    )


# ============================================================
# 5. Query Rewrite Decision
# ============================================================

class RewriteDecision(BaseModel):
    """
    Generates a better retrieval query when the current
    retrieval result is insufficient.
    """

    retrieval_query: str = Field(
        ...,
        description=(
            "Rewritten query optimized for semantic/vector "
            "retrieval against the internal documents."
        ),
    )




# ============================================================
# 7. Semantic Cache Metadata
# ============================================================

class CacheMetadata(TypedDict, total=False):
    """
    Metadata stored with a semantic-cache entry.
    """

    question: str
    answer: str
    evaluation: str
    cache_version: str

    