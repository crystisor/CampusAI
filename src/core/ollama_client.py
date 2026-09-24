import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from src.config import config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Asynchronous client for interacting with the local Ollama daemon."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or config.ollama.base_url).rstrip("/")
        self.timeout = timeout or config.ollama.request_timeout

    async def is_available(self) -> bool:
        """Check if Ollama server is running and reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        """List all models currently installed in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name", "") for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
        return []

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a complete text completion."""
        target_model = model or config.ollama.llm_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/api/generate", json=payload)
            res.raise_for_status()
            data = res.json()
            return data.get("response", "")

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens from Ollama."""
        target_model = model or config.ollama.llm_model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
                    except Exception:
                        continue

    async def get_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding vector for a single text string."""
        vectors = await self.get_embeddings([text], model=model)
        return vectors[0] if vectors else []

    async def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts using bge-m3."""
        target_model = model or config.ollama.embedding_model
        embeddings: List[List[float]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # First try the newer /api/embed batch endpoint
            try:
                res = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": target_model, "input": texts},
                )
                if res.status_code == 200:
                    data = res.json()
                    if "embeddings" in data:
                        return data["embeddings"]
            except Exception:
                pass

            # Fallback to /api/embeddings single item loop
            for text in texts:
                res = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": target_model, "prompt": text},
                )
                res.raise_for_status()
                data = res.json()
                embeddings.append(data.get("embedding", []))

        return embeddings
