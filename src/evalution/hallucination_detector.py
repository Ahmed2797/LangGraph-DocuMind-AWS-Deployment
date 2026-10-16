"""
=============================================================
Hallucination Detector — Detect fabricated claims
=============================================================
Uses an LLM-based evaluator to identify claims in the answer
that are NOT supported by the provided context.
"""

from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# =============================================================
# Output Schema
# =============================================================

class HallucinationResult(BaseModel):
    hallucination_score: float = Field(
        description="0.0 = no hallucination, 1.0 = fully hallucinated"
    )

    hallucinated_claims: List[str] = Field(
        default_factory=list,
        description="Claims in the answer that are not supported by context"
    )

    supported_claims: List[str] = Field(
        default_factory=list,
        description="Claims in the answer supported by the context"
    )

    reasoning: str = Field(
        description="Brief explanation of the evaluation"
    )


# =============================================================
# LLM
# =============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_llm = llm.with_structured_output(
    HallucinationResult
)


# =============================================================
# Prompt
# =============================================================

HALLUCINATION_PROMPT = """
You are a hallucination detection expert.

Your task is to determine whether the AI-generated answer contains
claims that are NOT supported by the provided context.

IMPORTANT:
The provided CONTEXT is the ONLY source of truth.

Do NOT use your own world knowledge.

If a claim may be true in the real world but is NOT supported by
the provided context, classify that claim as hallucinated.

CONTEXT:
{context}

AI ANSWER:
{answer}

RULES:

1. Break the answer into individual factual claims.

2. Check every claim against the context.

3. A claim is SUPPORTED if:
   - It is directly stated in the context, OR
   - It can be reasonably and safely inferred from the context.

4. A claim is HALLUCINATED if:
   - It is not present in the context, OR
   - It cannot reasonably be inferred from the context.

5. Do not assume missing information.

6. Do not use external or general knowledge.

7. hallucination_score:
   - 0.0 = no unsupported claims
   - 0.25 = very minor unsupported information
   - 0.5 = partially unsupported
   - 0.75 = mostly unsupported
   - 1.0 = fully unsupported

Return the structured evaluation.
"""


# =============================================================
# Detector
# =============================================================

def detect_hallucination(
    answer: str,
    context: str
) -> dict:
    """
    Detect hallucinations in an AI-generated answer.

    Args:
        answer: Generated answer text.
        context: Retrieved context used to generate the answer.

    Returns:
        Dictionary containing hallucination score,
        hallucinated claims, supported claims, and reasoning.
    """

    prompt = HALLUCINATION_PROMPT.format(
        context=context,
        answer=answer
    )

    try:

        result = structured_llm.invoke(prompt)

        return result.model_dump()

    except Exception as e:

        return {
            "hallucination_score": None,
            "hallucinated_claims": [],
            "supported_claims": [],
            "reasoning": f"Hallucination evaluation failed: {str(e)}",
        }