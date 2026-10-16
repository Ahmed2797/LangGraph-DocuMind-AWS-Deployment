from tavily import TavilyClient
import os
from src.config import get_settings


client = TavilyClient(
    api_key= get_settings().tavily_api_key
)


def tavily_search(query):
    response = client.search(
        query= query,
        max_results= 2
    )

    results = []

    for i, r in enumerate(response["results"], 1):
        title   = r.get("title", "Unknown")
        url     = r.get("url", "")
        snippet = r.get("content", "").strip()
        # Keep only the first 300 characters to avoid wall-of-text
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)