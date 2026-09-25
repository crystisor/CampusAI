import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.ollama_client import OllamaClient
from src.config import config

@pytest.mark.asyncio
async def test_ollama_generate_options_and_keep_alive():
    client = OllamaClient()
    
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test completion"}
        mock_post.return_value = mock_response

        # Test default LLM generate has keep_alive: -1 and think: False
        resp = await client.generate("Hello world")
        assert resp == "Test completion"
        assert mock_post.called
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload["model"] == config.ollama.llm_model
        assert payload["keep_alive"] == -1
        assert payload["think"] is False

@pytest.mark.asyncio
async def test_ollama_router_defaults_to_cpu():
    client = OllamaClient()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "RAG"}
        mock_post.return_value = mock_response

        # Test router model automatically gets num_gpu=0 if not specified
        resp = await client.generate("Route this", model=config.ollama.router_model)
        assert resp == "RAG"
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload["model"] == config.ollama.router_model
        assert payload["options"]["num_gpu"] == 0
        assert payload["keep_alive"] == config.ollama.router_keep_alive

@pytest.mark.asyncio
async def test_ollama_get_embeddings_cpu_pinning():
    client = OllamaClient()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_post.return_value = mock_response

        # Test embeddings default to num_gpu=0
        vecs = await client.get_embeddings(["Sample text"])
        assert len(vecs) == 1
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload["options"]["num_gpu"] == 0
        assert payload["keep_alive"] == config.ollama.embedding_keep_alive

@pytest.mark.asyncio
async def test_ollama_generate_think_parameter_override():
    client = OllamaClient()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Clean answer"}
        mock_post.return_value = mock_response

        # Test think override
        await client.generate("Hello", think=True)
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload["think"] is True

        await client.generate("Hello", think=False)
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload["think"] is False

@pytest.mark.asyncio
async def test_ollama_generate_sanitizes_thinking_and_closing_tag():
    client = OllamaClient()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulates model output where prompt template added <think> and model ended with </think>
        mock_response.json.return_value = {
            "response": "Let us consider the problem.\nDraft: 4.\nSo the answer is 4.</think>4"
        }
        mock_post.return_value = mock_response

        resp = await client.generate("What is 2+2?")
        # Must return ONLY the actual message without </think> or thought process
        assert resp == "4"
        assert "</think>" not in resp

@pytest.mark.asyncio
async def test_ollama_generate_400_fallback_without_think():
    client = OllamaClient()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # First call fails with 400 (unsupported think param on legacy Ollama)
        fail_response = MagicMock()
        fail_response.status_code = 400

        # Second call succeeds
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"response": "<think>thought</think>Answer"}

        mock_post.side_effect = [fail_response, success_response]

        resp = await client.generate("Test prompt")
        assert resp == "Answer"
        assert mock_post.call_count == 2
        # Verify second call omitted think parameter
        second_call_payload = mock_post.call_args_list[1][1]["json"]
        assert "think" not in second_call_payload

def test_clean_llm_response_various_formats():
    from src.core.ollama_client import clean_llm_response

    # 1. Closing </think> only (from template prompt prefix)
    t1 = "We need to explain what an eigenvalue is in detail...\nDraft 1: A scalar.\nSo output.</think>An eigenvalue is a scalar factor."
    assert clean_llm_response(t1) == "An eigenvalue is a scalar factor."

    # 2. Complete <think>...</think> tags
    t2 = "<think>Let me figure this out.\nDrafting message...</think>Here is the final response."
    assert clean_llm_response(t2) == "Here is the final response."

    # 3. Alternative tags: <thought>, <reasoning>, <draft>
    t3 = "<thought>Internal thought</thought>The answer is σ²."
    assert clean_llm_response(t3) == "The answer is σ²."

    t4 = "<reasoning>Step 1\nStep 2</reasoning>42"
    assert clean_llm_response(t4) == "42"

    t5 = "<draft>Initial draft</draft>Final text"
    assert clean_llm_response(t5) == "Final text"

    # 4. Markdown draft headers
    t6 = "**Thinking Process:**\nLet us compute 2+2.\n**Final Answer:**\nThe answer is 4."
    assert clean_llm_response(t6) == "The answer is 4."

    # 5. Clean text passes through unchanged
    t7 = "Calculus is the mathematical study of continuous change."
    assert clean_llm_response(t7) == t7

    # 6. Empty / None
    assert clean_llm_response("") == ""
    assert clean_llm_response(None) == ""
