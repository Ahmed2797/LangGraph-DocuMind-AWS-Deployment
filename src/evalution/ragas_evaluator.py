"""
=============================================================
RAGAS Evaluator — Offline evaluation using RAGAS metrics
=============================================================
Computes faithfulness, answer_relevancy, context_precision,
and context_recall for benchmarking and dashboard reporting.

NOTE: This module is for OFFLINE evaluation only.
Do NOT use it in the real-time query pipeline.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path

from src.config import METRICS_DIR
from src.logger import logging

logger = logging()


def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict:
    """
    Run RAGAS evaluation on a single QA pair.
    
    Args:
        question: User question
        answer: Generated answer
        contexts: List of context strings used for generation
        ground_truth: Optional ground truth answer
    
    Returns:
        Dict with faithfulness, answer_relevancy, context_precision, context_recall
    """
    logger.info("Running RAGAS evaluation...")

    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset

        # Build dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        # Select metrics
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truth:
            metrics.append(context_recall)

        result = ragas_evaluate(dataset=dataset, metrics=metrics)
        scores = result.to_pandas().iloc[0].to_dict()

        # Clean up scores
        output = {}
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if key in scores:
                val = scores[key]
                output[key] = round(float(val), 4) if val is not None else None

        logger.info(f"RAGAS scores: {output}")
        return output

    except ImportError:
        logger.warning("RAGAS library not installed. Install with: pip install ragas")
        return _fallback_evaluation(question, answer, contexts)
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return _fallback_evaluation(question, answer, contexts)


def _fallback_evaluation(question: str, answer: str, contexts: List[str]) -> Dict:
    """Simple heuristic fallback when RAGAS is unavailable."""
    logger.info("Using fallback evaluation (RAGAS unavailable)")

    context_text = " ".join(contexts).lower()
    answer_lower = answer.lower()
    question_lower = question.lower()

    # Simple word overlap heuristics
    answer_words = set(answer_lower.split())
    context_words = set(context_text.split())
    question_words = set(question_lower.split())
    
    # Filter stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "but", "with"}
    answer_words -= stop_words
    context_words -= stop_words
    question_words -= stop_words

    # Faithfulness: how many answer words appear in context
    if answer_words:
        faithfulness = len(answer_words & context_words) / len(answer_words)
    else:
        faithfulness = 0.0

    # Answer relevancy: how many question words appear in answer
    if question_words:
        relevancy = len(question_words & answer_words) / len(question_words)
    else:
        relevancy = 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "context_precision": None,
        "context_recall": None,
    }


def run_batch_evaluation(
    qa_pairs: List[Dict],
    output_path: Path = None,
) -> List[Dict]:
    """
    Run RAGAS evaluation on a batch of QA pairs.
    
    Args:
        qa_pairs: List of dicts with keys: question, answer, contexts, ground_truth (optional)
        output_path: Path to save results JSON
    
    Returns:
        List of evaluation result dicts
    """
    if output_path is None:
        output_path = METRICS_DIR / "ragas_results.json"

    results = []
    for i, pair in enumerate(qa_pairs):
        logger.info(f"Evaluating pair {i+1}/{len(qa_pairs)}")
        result = evaluate_with_ragas(
            question=pair["question"],
            answer=pair["answer"],
            contexts=pair.get("contexts", []),
            ground_truth=pair.get("ground_truth"),
        )
        result["question"] = pair["question"]
        results.append(result)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"RAGAS results saved to {output_path}")

    return results


