import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx
from duckduckgo_search import DDGS
from src.config import config
from src.core.reranker import RerankerCandidate

logger = logging.getLogger(__name__)

class WebSearchEngine:
    """Async web search provider using DuckDuckGo or Tavily."""

    def __init__(self, provider: Optional[str] = None, max_results: Optional[int] = None):
        self.provider = provider or config.search.provider
        self.max_results = max_results or config.search.max_results
        self.tavily_api_key = config.search.tavily_api_key

    async def search(self, query: str, max_results: Optional[int] = None) -> List[RerankerCandidate]:
        """Execute web search and return formatted RerankerCandidate objects."""
        k = max_results or self.max_results

        if self.provider == "tavily" and self.tavily_api_key:
            return await self._search_tavily(query, k)
        else:
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

    async def _search_tavily(self, query: str, max_results: int) -> List[RerankerCandidate]:
        """Tavily search API integration."""
        candidates = []
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "basic",
                        "max_results": max_results,
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    for item in data.get("results", []):
                        title = item.get("title", "")
                        content = item.get("content", "")
                        url = item.get("url", "")
                        candidates.append(
                            RerankerCandidate(
                                text=f"**{title}**\n{content}\n*Source: {url}*",
                                source="web_search",
                                metadata={"title": title, "url": url},
                                score=float(item.get("score", 0.6))
                            )
                        )
        except Exception as e:
            logger.error(f"Tavily search error: {e}. Falling back to DuckDuckGo.")
            return await self._search_ddg(query, max_results)

        return candidates
