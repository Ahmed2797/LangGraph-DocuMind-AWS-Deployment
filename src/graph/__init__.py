"""
RagVision — LangGraph workflow builder
======================================

Use ainvoke / astream (web_search is an async node).
Everything per-turn is reset inside `prepare_query`, so you only pass
`question` (+ optional `query_image`).
"""
import sys
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.exception import CustomException
from src.logger import logging
from src.retrive.retrieval_pipeline import (
    build_retrieve_node,
    check_grounded,
    check_retrieval_score,
    check_semantic_cache_node,
    check_useful,
    evaluate_answer_node,
    finalize_response,
    generate_direct,
    generate_from_context,
    no_answer_found,
    optimizer_retrieval_node,
    prepare_query,
    revise_answer,
    rewrite_question,
    route_after_cache,
    route_after_grounded,
    route_after_retrieval_score,
    route_after_useful,
    route_by_query_type,
    route_query,
    save_cache_node,
    web_search_node,
)
from src.state import State

# Node id -> label for the frontend's animated graph view.
NODE_LABELS = {
    "prepare_query": "Prepare Query (image + history)",
    "check_semantic_cache": "Check Semantic Cache",
    "route_query": "Route Query (RAG / Web / Direct)",
    "retrieve": "Retrieve from Vector DB",
    "check_retrieval_score": "Filter by Similarity",
    "query_optimizer": "Optimize Query",
    "generate_from_context": "Generate from Context",
    "check_grounded": "Groundedness Check",
    "revise_answer": "Revise with Quotes",
    "check_useful": "Usefulness Check",
    "rewrite_question": "Rewrite Query",
    "evaluate_answer": "Evaluate Answer",
    "save_cache": "Save to Semantic Cache",
    "web_search": "Web Search (text + images)",
    "generate_direct": "Direct Answer",
    "no_answer_found": "No Answer Found",
    "finalize": "Finalize Response",
}


def create_graph(
    retriever,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    project_id: Optional[str] = None,
    top_k: int = 5,
):
    """
    Build and compile the RagVision workflow with multi-turn memory.

    retriever    : FAISS / Pinecone vector store (must support similarity_search_with_score)
    project_id   : used as the semantic-cache namespace (cache is per project)
    """
    ns = project_id
    try:
        logging.info("Building LangGraph.")
        g = StateGraph(State)
        checkpointer = checkpointer or MemorySaver()

        # ---------------- nodes ----------------
        g.add_node("prepare_query", prepare_query)
        g.add_node("check_semantic_cache", lambda s: check_semantic_cache_node(s, namespace=ns))
        g.add_node("route_query", route_query)

        g.add_node("retrieve", build_retrieve_node(retriever, k=top_k))
        g.add_node("check_retrieval_score", check_retrieval_score)
        g.add_node("query_optimizer", optimizer_retrieval_node)
        g.add_node("generate_from_context", generate_from_context)
        g.add_node("check_grounded", check_grounded)
        g.add_node("revise_answer", revise_answer)
        g.add_node("check_useful", check_useful)
        g.add_node("rewrite_question", rewrite_question)
        g.add_node("evaluate_answer", evaluate_answer_node)
        g.add_node("save_cache", lambda s: save_cache_node(s, namespace=ns))

        g.add_node("web_search", web_search_node)
        g.add_node("generate_direct", generate_direct)
        g.add_node("no_answer_found", no_answer_found)
        g.add_node("finalize", finalize_response)

        # ---------------- edges ----------------
        g.add_edge(START, "prepare_query")
        g.add_edge("prepare_query", "check_semantic_cache")

        g.add_conditional_edges(
            "check_semantic_cache",
            route_after_cache,
            {"cached_answer": "finalize", "continue_rag": "route_query"},
        )

        g.add_conditional_edges(
            "route_query",
            route_by_query_type,
            {
                "retrieve": "retrieve",
                "web_search": "web_search",
                "generate_direct": "generate_direct",
            },
        )

        # retrieval loop
        g.add_edge("retrieve", "check_retrieval_score")
        g.add_conditional_edges(
            "check_retrieval_score",
            route_after_retrieval_score,
            {
                "generate_from_context": "generate_from_context",
                "query_optimizer": "query_optimizer",
                "web_search": "web_search",        # only used if WEB_FALLBACK_ON_RETRIEVAL_FAIL = True
                "no_answer_found": "no_answer_found",
            },
        )
        g.add_edge("query_optimizer", "retrieve")

        # self-correction
        g.add_edge("generate_from_context", "check_grounded")
        g.add_conditional_edges(
            "check_grounded",
            route_after_grounded,
            {
                "accept_answer": "check_useful",
                "revise_answer": "revise_answer",
                "no_answer_found": "no_answer_found",
            },
        )
        g.add_edge("revise_answer", "check_grounded")

        g.add_conditional_edges(
            "check_useful",
            route_after_useful,
            {
                "evaluate_answer": "evaluate_answer",
                "rewrite_question": "rewrite_question",
                "no_answer_found": "no_answer_found",
            },
        )
        g.add_edge("rewrite_question", "retrieve")

        # finish
        g.add_edge("evaluate_answer", "save_cache")
        g.add_edge("save_cache", "finalize")
        g.add_edge("web_search", "finalize")
        g.add_edge("generate_direct", "finalize")
        g.add_edge("no_answer_found", "finalize")
        g.add_edge("finalize", END)

        app = g.compile(checkpointer=checkpointer)
        logging.info("LangGraph compiled.")
        return app

    except Exception as e:
        logging.error(f"Error while creating LangGraph: {e}")
        raise CustomException(e, sys)

    