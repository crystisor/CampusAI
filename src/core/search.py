import asyncio
import logging
from typing import List, Optional
from duckduckgo_search import DDGS
from src.config import config
from src.core.reranker import RerankerCandidate

logger = logging.getLogger(__name__)

class WebSearchEngine:
    """Async web search provider using DuckDuckGo."""

    def __init__(self, max_results: Optional[int] = None):
        self.max_results = max_results or config.search.max_results

    async def search(self, query: str, max_results: Optional[int] = None) -> List[RerankerCandidate]:
        """Execute web search and return formatted RerankerCandidate objects."""
        k = max_results or self.max_results

        return await self._search_ddg(query, k)

    async def _search_ddg(self, query: str, max_results: int) -> List[RerankerCandidate]:
        """DuckDuckGo search running in a threadpool to prevent blocking the event loop."""
        loop = asyncio.get_running_loop()

        def _sync_ddg():
            results = []
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=max_results))
                    for item in raw:
                        title = item.get("title", "")
                        body = item.get("body", "")
                        href = item.get("href", "")
                        formatted_text = f"**{title}**\n{body}\n*Source: {href}*"
                        results.append(
                            RerankerCandidate(
                                text=formatted_text,
                                source="web_search",
                                metadata={"title": title, "url": href},
                                score=0.5
                            )
                        )
            except Exception as e:
                logger.error(f"DuckDuckGo search error: {e}")
            return results

        return await loop.run_in_executor(None, _sync_ddg)
