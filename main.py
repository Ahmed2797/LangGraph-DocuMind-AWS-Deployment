import time
import asyncio

from src.pipeline import pipeline
from langchain_community.callbacks import get_openai_callback
from src.logger.production_logger import ProductionLogger


production_logger = ProductionLogger(
    log_file="logs/rag_production.log"
)

config = {
    "configurable": {
        "thread_id": "user_session_101"
    }
}


async def run_rag(question: str):
    """Run the asynchronous RAG pipeline and log execution metrics."""

    initial_state = {
        "question": question,
        "retrieval_query": "",
        "optimized_query": "",
        "keywords": [],
        "entities": [],
        "rewrite_tries": 2,
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "",
        "evidence": [],
        "retries": 1,
        "isuse": "",
        "evaluation": {},
        "cache_hit": False,
        "need_retrieval": None,
    }

    start_time = time.perf_counter()

    try:
        async with pipeline("test_run") as app:
            with get_openai_callback() as callback:
                final_state = await app.ainvoke(
                    initial_state,
                    config=config,
                )

                latency_ms = round(
                    (time.perf_counter() - start_time) * 1000,
                    2,
                )

                metrics = {
                    "Prompt tokens": callback.prompt_tokens,
                    "Completion tokens": callback.completion_tokens,
                    "latency_ms": latency_ms,
                    "tokens_used": callback.total_tokens,
                    "query_cost": callback.total_cost,
                }

        production_logger.log_query(
            state=final_state,
            metrics=metrics,
        )

        return final_state

    except Exception:
        raise


if __name__ == "__main__":
    final_state = asyncio.run(
        run_rag(question="right now upcoming holiday in italy")
    )

    print("Final State:", final_state)
    print("*" * 50)
    print("\nAnswer:", final_state["answer"])
    