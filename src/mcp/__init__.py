import os
import certifi
from asyncio import Lock

from src.config import get_settings
from langchain_mcp_adapters.client import MultiServerMCPClient


# ==========================================
# Environment configuration
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = get_settings().tavily_api_key


# ==========================================
# MCP client configuration
# ==========================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY}"
            ),
        }
    }
)


# ==========================================
# Global tool state
# ==========================================

search_tool = None
_initialization_lock = Lock()


# ==========================================
# Initialize Tavily MCP
# ==========================================

async def initialize_mcp():
    """
    Initialize the Tavily MCP server and load the
    tavily_search tool.

    The initialization is performed only once.
    """

    global search_tool

    if search_tool is not None:
        return

    async with _initialization_lock:

        if search_tool is not None:
            return

        try:
            tools = await client.get_tools(
                server_name="tavily"
            )

            tools_by_name = {
                tool.name: tool
                for tool in tools
            }

            search_tool = tools_by_name.get(
                "tavily_search"
            )

            if search_tool is None:
                available_tools = ", ".join(
                    tools_by_name.keys()
                )

                raise RuntimeError(
                    "Tavily MCP connected, but "
                    "'tavily_search' was not found. "
                    f"Available tools: "
                    f"{available_tools or 'none'}"
                )

        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize Tavily MCP: {error}"
            ) from error


# ==========================================
# Tavily search
# ==========================================

async def tavily_mcp_search(query: str):
    """
    Search the web using Tavily MCP.

    Args:
        query: Search query.

    Returns:
        Tavily MCP search result.

    Raises:
        ValueError: If the query is empty.
        RuntimeError: If Tavily MCP search fails.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    await initialize_mcp()

    try:
        result = await search_tool.ainvoke(
            {
                "query": query.strip()
            }
        )

        return result

    except Exception as error:
        raise RuntimeError(
            f"Tavily MCP search failed: {error}"
        ) from error

