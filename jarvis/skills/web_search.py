"""Web search skill using DuckDuckGo (no API key needed)"""

from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results as string
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}\n   {r['href']}\n   {r['body'][:200]}...")

        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search failed: {str(e)}"
