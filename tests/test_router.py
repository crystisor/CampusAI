import pytest
from unittest.mock import AsyncMock
from src.core.router import IntentRouter, RoutingDecision

@pytest.mark.asyncio
async def test_router_fallback():
    mock_ollama = AsyncMock()
    # Mock fallback response when ollama returns valid intent json
    mock_ollama.generate.return_value = '{"decision": "RAG", "reason": "Query asks about lecture slides"}'

    router = IntentRouter(mock_ollama)
    decision, reason = await router.route("What did the professor say about eigenvectors in lecture 3?", subject_name="Linear Algebra")

    assert decision == RoutingDecision.RAG
    assert "lecture" in reason.lower()
    # Verify num_gpu=0 was passed in options
    mock_ollama.generate.assert_called_once()
    _, kwargs = mock_ollama.generate.call_args
    assert kwargs.get("options", {}).get("num_gpu") == 0

@pytest.mark.asyncio
async def test_router_heuristic_fallback():
    mock_ollama = AsyncMock()
    # Mock ollama failing/timing out
    mock_ollama.generate.side_effect = Exception("Ollama offline")

    router = IntentRouter(mock_ollama)
    
    # Direct test
    dec_direct, _ = await router.route("Hi, how are you?")
    assert dec_direct == RoutingDecision.DIRECT

    # Web test
    dec_web, _ = await router.route("What is the latest news today?")
    assert dec_web == RoutingDecision.WEB

    # RAG test
    dec_rag, _ = await router.route("In lecture 2, what was the theorem proven?")
    assert dec_rag == RoutingDecision.RAG
