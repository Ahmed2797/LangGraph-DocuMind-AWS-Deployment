"""
=============================================================
Groundedness Scorer — Measure how well-grounded an answer is
=============================================================
Evaluates how accurately an AI-generated answer is supported
by the provided context.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# =============================================================
# Output Schema
# =============================================================

class GroundednessResult(BaseModel):

    groundedness_score: float = Field(
        description="Overall groundedness score from 0.0 to 1.0"
    )

    evidence_alignment: float = Field(
        description="How well the answer's claims are supported by the context"
    )

    completeness: float = Field(
        description="How completely the answer addresses the question using available context"
    )

    accuracy: float = Field(
        description="How factually consistent the answer is with the provided context"
    )

    relevance: float = Field(
        description="How relevant the answer is to the user's question"
    )

    reasoning: str = Field(
        description="Brief explanation of the scores"
    )


# =============================================================
# LLM
# =============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

structured_llm = llm.with_structured_output(
    GroundednessResult
)


# =============================================================
# Prompt
# =============================================================

GROUNDEDNESS_PROMPT = """
You are an expert evaluator of RAG answer groundedness.

Your task is to evaluate how well the AI-generated answer is
supported by the provided context.

IMPORTANT:
The provided CONTEXT is the ONLY source of truth.

Do NOT use your own world knowledge.

A statement may be factually true in the real world, but if it
is NOT supported by the provided context, it must be considered
NOT GROUNDED.

------------------------------------------------------------
CONTEXT:
{context}
------------------------------------------------------------

QUESTION:
{question}

------------------------------------------------------------

AI ANSWER:
{answer}
------------------------------------------------------------

Evaluate the answer using these dimensions:

1. Evidence Alignment
   - Are the claims supported by evidence in the context?
   - Unsupported claims should lower this score.

2. Completeness
   - Does the answer sufficiently address the question?
   - Consider only information that is available in the context.
   - Do not penalize the answer for information that does not exist
     in the context.

3. Accuracy
   - Are the claims consistent with the context?
   - If a claim contradicts or is unsupported by the context,
     reduce the accuracy score.

4. Relevance
   - Does the answer directly address the question?
   - Avoid rewarding unrelated information.

5. Groundedness Score
   - Overall measure of how strongly the answer is supported
     by the provided context.

Scoring:

0.0 = completely unsupported
0.25 = mostly unsupported
0.50 = partially supported
0.75 = mostly supported
1.0 = fully supported

IMPORTANT CONSISTENCY RULE:

If a factual claim is unsupported by the context:

- Evidence alignment must decrease.
- Accuracy must decrease.
- Groundedness must decrease.

Do NOT give accuracy = 1.0 to an unsupported claim.

Return the structured evaluation.
"""


# =============================================================
# Groundedness Evaluator
# =============================================================

def score_groundedness(
    question: str,
    answer: str,
    context: str
) -> dict:
    """
    Score how well an answer is grounded in the provided context.

    Args:
        question: User's question.
        answer: Generated AI answer.
        context: Retrieved context.

    Returns:
        Dictionary containing groundedness score and sub-scores.
    """

    prompt = GROUNDEDNESS_PROMPT.format(
        question=question,
        answer=answer,
        context=context
    )

    try:

        result = structured_llm.invoke(prompt)

        return result.model_dump()

    except Exception as e:

        return {
            "groundedness_score": None,
            "evidence_alignment": None,
            "completeness": None,
            "accuracy": None,
            "relevance": None,
            "reasoning": f"Groundedness evaluation failed: {str(e)}",
        }