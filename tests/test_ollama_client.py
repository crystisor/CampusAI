import json
import asyncio
from contextlib import asynccontextmanager, contextmanager

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from src.core.ollama_client import OllamaClient
from src.config import config


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["done", "incomplete", "error", "deadline", "idle"])
async def test_chat_stream_progress_and_failures(monkeypatch, ending):
    closed = False

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"message":{"thinking":"private reasoning"},"done":false}\n'
            yield b'{"message":{"content":"Final "},"done":false}\n'
            if ending == "deadline":
                await asyncio.sleep(10)
            elif ending == "idle":
                raise httpx.ReadTimeout("No progress")
            elif ending == "error":
                yield b'{"error":"model failed"}\n'
            elif ending == "done":
                yield b'{"message":{"content":"answer"},"done":true}\n'

        async def aclose(self):
            nonlocal closed
            closed = True

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        assert request.url.path == "/api/chat"
        return httpx.Response(200, stream=Stream())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(config.ollama, "generation_timeout", 0.1)
    if ending == "done":
        assert await OllamaClient().generate("question") == "Final answer"
    else:
        error = {"deadline": TimeoutError, "idle": httpx.ReadTimeout}.get(ending, RuntimeError)
        with pytest.raises(error):
            await OllamaClient().generate("question")
    assert closed


@contextmanager
def mock_chat_stream():
    """Adapt a complete fixture response to the chat NDJSON wire format."""
    request = AsyncMock()

    @asynccontextmanager
    async def stream(client, method, url, **kwargs):
        response = await request(url, **kwargs)

        async def lines():
            yield json.dumps({**response.json(), "done": True})

        response.aiter_lines = lines
        yield response

    with patch("httpx.AsyncClient.stream", new=stream):
        yield request

@pytest.mark.asyncio
async def test_ollama_generate_options_and_keep_alive(monkeypatch):
    monkeypatch.setattr(config.ollama, "llm_num_ctx", 6144)
    client = OllamaClient()
    
    with mock_chat_stream() as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Test completion"}}
        mock_post.return_value = mock_response

        # Default generation keeps the LLM loaded and uses configured thinking.
        resp = await client.generate("Hello world")
        assert resp == "Test completion"
        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0].endswith("/api/chat")
        payload = kwargs.get("json", {})
        assert payload["stream"] is True
        assert payload["model"] == config.ollama.llm_model
        assert payload["messages"] == [{"role": "user", "content": "Hello world"}]
        assert "prompt" not in payload
        assert payload["keep_alive"] == -1
        assert payload["think"] is config.ollama.think
        assert payload["options"] == {
            "temperature": 0.2,
            "num_ctx": 6144,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "num_predict": config.ollama.llm_max_tokens,
        }

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
        assert payload["options"]["temperature"] == 0.0
        assert payload["options"]["num_predict"] == 10
        assert "max_tokens" not in payload["options"]
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

    with mock_chat_stream() as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "Clean answer"}}
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

    with mock_chat_stream() as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Simulates model output where prompt template added <think> and model ended with </think>
        mock_response.json.return_value = {
            "message": {"content": "Let us consider the problem.\nDraft: 4.\nSo the answer is 4.</think>4"}
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

        resp = await client.generate("Test prompt", model=config.ollama.router_model)
        assert resp == "Answer"
        assert mock_post.call_count == 2
        # Verify second call omitted think parameter
        second_call_payload = mock_post.call_args_list[1][1]["json"]
        assert "think" not in second_call_payload

@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Supervised learning uses labeled examples.", "", None])
async def test_chat_never_returns_drafting_as_answer(content):
    client = OllamaClient()
    draft = "The user is asking about supervised learning. Let's outline the answer."
    with mock_chat_stream() as mock_post:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "message": {"role": "assistant", "thinking": draft, "content": content},
            "response": draft,
            "done_reason": "stop" if content else "length",
        }
        mock_post.return_value = response
        result = await client.generate("Explain supervised learning", system="Teach clearly.", think=True)
        assert result == (content or "")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["think"] is True
        assert payload["messages"] == [
            {"role": "system", "content": "Teach clearly."},
            {"role": "user", "content": "Explain supervised learning"},
        ]
        assert "system" not in payload


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
