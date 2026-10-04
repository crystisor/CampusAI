from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.reranker import RerankerCandidate
from src.rag.pipeline import RAGPipeline
from src.rag.qdrant_manager import QdrantManager


@pytest.mark.asyncio
async def test_caption_lookup_scans_pages_and_excludes_mentions_and_other_numbers():
    manager = QdrantManager.__new__(QdrantManager)
    manager.prefix = "subject_"
    def point(text):
        return SimpleNamespace(payload={"text": text, "document_name": "Article", "page_number": 7})
    manager.client = Mock(scroll=AsyncMock(side_effect=[
        ([point("Table 2 summarizes the heuristics."), point("TABLE 20: Other")], "next"),
        ([point("## TABLE 2: Development cycles\nTest-first ...")], None),
    ]))
    result = await manager.find_table("articles", "2")
    assert len(result) == 1
    assert result[0].metadata["page_number"] == 7
    assert manager.client.scroll.call_args.kwargs["offset"] == "next"
    assert manager.client.scroll.call_args.kwargs["collection_name"] == "subject_articles"


def pipeline_with_tables(documents):
    candidates = [RerankerCandidate("TABLE 2: Development cycles", "course_material",
                  {"document_name": name, "page_number": 7}) for name in documents]
    return RAGPipeline(
        ollama_client=Mock(generate=AsyncMock(return_value="Table explanation"), get_embedding=AsyncMock()),
        router=Mock(route=AsyncMock()), reranker=Mock(),
        qdrant_manager=Mock(find_table=AsyncMock(return_value=candidates)), search_engine=Mock(),
    )


@pytest.mark.asyncio
async def test_exact_caption_reaches_answer_without_semantic_rejection():
    pipeline = pipeline_with_tables(["Article"])
    result = await pipeline.process_query("explain table 2 from the article", "articles")
    assert result["answer"] == "Table explanation"
    assert result["top_contexts"][0]["metadata"]["page_number"] == 7
    pipeline.router.route.assert_not_awaited()
    pipeline.ollama.get_embedding.assert_not_awaited()
    pipeline.reranker.rerank.assert_not_called()


@pytest.mark.asyncio
async def test_ambiguous_table_requests_document_instead_of_guessing():
    pipeline = pipeline_with_tables(["Article A", "Article B"])
    result = await pipeline.process_query("explain table 2", "articles")
    assert "Article A" in result["answer"] and "Article B" in result["answer"]
    pipeline.ollama.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_named_document_resolves_duplicate_captions():
    pipeline = pipeline_with_tables(["Article A", "Article B"])
    result = await pipeline.process_query("explain table 2 in Article B", "articles")
    assert [c["metadata"]["document_name"] for c in result["top_contexts"]] == ["Article B"]


@pytest.mark.asyncio
async def test_missing_caption_falls_back_to_existing_retrieval():
    from src.core.router import RoutingDecision
    pipeline = pipeline_with_tables([])
    pipeline.router.route.return_value = (RoutingDecision.RAG, "RAG")
    pipeline.ollama.get_embedding.return_value = [0.1]
    pipeline.qdrant.search = AsyncMock(return_value=[])
    result = await pipeline.process_query("explain table 99", "articles")
    pipeline.qdrant.search.assert_awaited_once()
    assert "couldn't retrieve" in result["answer"]
