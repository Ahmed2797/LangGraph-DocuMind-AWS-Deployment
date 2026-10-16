import sys
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from src.exception import CustomException
from src.graph import create_graph
from src.logger import logging
from src.retrive import get_project_retriever


@asynccontextmanager
async def pipeline(project_id: str):
    """
    Create and manage the asynchronous LangGraph RAG pipeline.

    Args:
        project_id: Identifier for the document project.

    Yields:
        CompiledStateGraph: Compiled LangGraph application.

    Raises:
        CustomException: If retriever or graph creation fails.

    Usage:
        async with pipeline(project_id) as app:
            config = {"configurable": {"thread_id": f"{project_id}:{chat_id}"}}
            async for chunk in app.astream(
                {"question": q, "query_image": image_or_none},
                config,
                stream_mode="updates",
            ):
                ...   # chunk = {node_name: state_update}
    """
    formatted_project_id = str(project_id).lower().replace("_", "-")

    # Setup errors are wrapped in CustomException; errors raised by the caller's
    # own `async with` body pass through unchanged.
    async with AsyncExitStack() as stack:
        try:
            logging.info(f"Starting RAG pipeline for project: {formatted_project_id}")
            retriever = get_project_retriever(formatted_project_id)
            logging.info("Retriever created successfully.")

            memory = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string("checkpoints.sqlite")
            )
            # async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
            app = create_graph(
                retriever=retriever,
                checkpointer=memory,
                project_id=formatted_project_id,   # semantic cache is scoped per project
            )
            logging.info("LangGraph application created successfully.")
        except Exception as e:
            logging.error(f"Error in RAG pipeline setup: {e}")
            raise CustomException(e, sys)

        yield app