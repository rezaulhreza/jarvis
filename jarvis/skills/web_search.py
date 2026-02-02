"""Web search skill using DuckDuckGo (no API key needed)"""

from ddgs import DDGS


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


def get_current_news(topic: str) -> str:
    """
    Get current news and recent information about a topic.
    Use this for questions about current events, recent news, or anything that
    requires up-to-date information (politics, sports, technology, celebrities, etc).

    Args:
        topic: The topic to search for current news about

    Returns:
        Recent news and information about the topic
    """
    try:
        with DDGS() as ddgs:
            # Search news specifically
            results = list(ddgs.news(topic, max_results=5))

        if not results:
            # Fallback to regular search with date qualifier
            return web_search(f"{topic} 2026 latest news", max_results=5)

        formatted = [f"**Latest news about: {topic}**\n"]
        for i, r in enumerate(results, 1):
            date = r.get('date', 'Recent')
            formatted.append(
                f"{i}. {r['title']}\n"
                f"   Date: {date}\n"
                f"   Source: {r.get('source', 'Unknown')}\n"
                f"   {r['body'][:200]}...\n"
                f"   URL: {r['url']}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        # Fallback to regular search
        return web_search(f"{topic} 2026 latest", max_results=5)
