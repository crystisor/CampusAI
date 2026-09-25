import re
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from src.config import config

logger = logging.getLogger(__name__)

def clean_llm_response(text: str) -> str:
    """
    Strips internal reasoning tokens, thinking process tags (<think>...</think>, <thought>, etc.),
    internal monologues, and draft blocks from LLM output so that only the final,
    actual user-facing message is returned.
    """
    if not text:
        return ""

    # 1. Remove complete XML-style thinking/reasoning/draft blocks
    pattern = r"<(?P<tag>think|thought|reasoning|reflection|scratchpad|draft)\b[^>]*>.*?</(?P=tag)>"
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. In models where <think> was injected into the prompt template, the response begins
    # directly with thinking tokens and ends with </think>. Everything prior to the last
    # closing think/thought/reasoning tag is part of the internal thinking process.
    cleaned = re.sub(
        r"^.*</(?:think|thought|reasoning|reflection|scratchpad|draft)>\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Handle unclosed thinking blocks (e.g. if generation terminated early inside thinking)
    cleaned = re.sub(
        r"<(?P<tag>think|thought|reasoning|reflection|scratchpad|draft)\b[^>]*>.*$",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 4. Remove any residual opening/closing thinking or draft tags
    cleaned = re.sub(
        r"</?(?:think|thought|reasoning|reflection|scratchpad|draft)>",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 5. Remove any markdown-style thinking/draft headers followed by "Final Answer" or "Response"
    cleaned = re.sub(
        r"^(?:[#*~_=\s-]*)(?:thinking process|internal monologue|reasoning|drafts?)\b.*?(?:final answer|actual response|response)\b[:\*#~_=\s-]*\n*\s*",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return cleaned.strip()

class OllamaClient:
    """Asynchronous client for interacting with the local Ollama daemon."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = (base_url or config.ollama.base_url).rstrip("/")
        self.timeout = timeout or config.ollama.request_timeout

    @staticmethod
    def _clean_keep_alive(val: Any) -> Any:
        """Converts '-1' string duration to integer -1 so Ollama accepts indefinite retention."""
        if val is None:
            return None
        if isinstance(val, str) and val.strip() == "-1":
            return -1
        return val

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
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> str:
        """Generate a complete text completion with thinking process filtered out."""
        target_model = model or config.ollama.llm_model
        
        # Determine appropriate keep_alive
        if keep_alive is None:
            if target_model == config.ollama.llm_model:
                keep_alive = config.ollama.llm_keep_alive
            elif target_model == config.ollama.router_model:
                keep_alive = config.ollama.router_keep_alive

        merged_options = dict(options or {})
        if target_model == config.ollama.router_model and "num_gpu" not in merged_options:
            merged_options["num_gpu"] = config.ollama.router_num_gpu

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if merged_options:
            payload["options"] = merged_options
        cleaned_keep_alive = self._clean_keep_alive(keep_alive)
        if cleaned_keep_alive is not None:
            payload["keep_alive"] = cleaned_keep_alive

        # Thinking configuration (defaults to False to avoid thinking/draft output)
        use_think = think if think is not None else getattr(config.ollama, "think", False)
        payload["think"] = use_think

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/api/generate", json=payload)
            if res.status_code == 400 and "think" in payload:
                # If an older Ollama daemon does not support the 'think' parameter, retry without it
                fallback_payload = dict(payload)
                fallback_payload.pop("think", None)
                res = await client.post(f"{self.base_url}/api/generate", json=fallback_payload)
            res.raise_for_status()
            data = res.json()
            raw_response = data.get("response", "")
            return clean_llm_response(raw_response)

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens from Ollama without thinking process."""
        target_model = model or config.ollama.llm_model

        if keep_alive is None:
            if target_model == config.ollama.llm_model:
                keep_alive = config.ollama.llm_keep_alive
            elif target_model == config.ollama.router_model:
                keep_alive = config.ollama.router_keep_alive

        merged_options = dict(options or {})
        if target_model == config.ollama.router_model and "num_gpu" not in merged_options:
            merged_options["num_gpu"] = config.ollama.router_num_gpu

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if merged_options:
            payload["options"] = merged_options
        cleaned_keep_alive = self._clean_keep_alive(keep_alive)
        if cleaned_keep_alive is not None:
            payload["keep_alive"] = cleaned_keep_alive

        use_think = think if think is not None else getattr(config.ollama, "think", False)
        payload["think"] = use_think

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        import json
                        chunk = json.loads(line)
                        # Only yield final response content, omit thinking field
                        token = chunk.get("response", "")
                        if token and "</think>" not in token:
                            yield token
                        elif token and "</think>" in token:
                            # Token contains </think>, yield only text after </think>
                            post_think = token.split("</think>")[-1]
                            if post_think:
                                yield post_think
                    except Exception:
                        continue

    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        num_gpu: Optional[int] = None,
        keep_alive: Optional[str] = None,
    ) -> List[float]:
        """Generate embedding vector for a single text string."""
        vectors = await self.get_embeddings(
            [text],
            model=model,
            num_gpu=num_gpu,
            keep_alive=keep_alive,
        )
        return vectors[0] if vectors else []

    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        num_gpu: Optional[int] = None,
        keep_alive: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts using bge-m3 (CPU-pinned by default)."""
        target_model = model or config.ollama.embedding_model
        gpu_count = num_gpu if num_gpu is not None else config.ollama.embedding_num_gpu
        alive = self._clean_keep_alive(keep_alive if keep_alive is not None else config.ollama.embedding_keep_alive)
        embeddings: List[List[float]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # First try the newer /api/embed batch endpoint
            try:
                res = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": target_model,
                        "input": texts,
                        "keep_alive": alive,
                        "options": {"num_gpu": gpu_count},
                    },
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
                    json={
                        "model": target_model,
                        "prompt": text,
                        "keep_alive": alive,
                        "options": {"num_gpu": gpu_count},
                    },
                )
                res.raise_for_status()
                data = res.json()
                embeddings.append(data.get("embedding", []))

        return embeddings

    async def warm_models(self) -> Dict[str, bool]:
        """
        Pre-warms models into their designated memory tiers:
        - LLM (Spark-X2.5-4b) into GPU VRAM (keep_alive: -1)
        - Arch-Router into CPU RAM (num_gpu: 0, keep_alive: 30m)
        - bge-m3 into CPU RAM (num_gpu: 0, keep_alive: 30m)
        """
        logger.info("Pre-warming models for zero-latency Discord operation...")
        status = {"llm": False, "router": False, "embedding": False}

        # 1. Warm LLM on GPU
        try:
            logger.info(f"Warming LLM '{config.ollama.llm_model}' on GPU...")
            await self.generate(
                prompt="Hello",
                model=config.ollama.llm_model,
                options={"max_tokens": 1},
                keep_alive=config.ollama.llm_keep_alive,
                think=False,
            )
            status["llm"] = True
            logger.info(f"LLM '{config.ollama.llm_model}' pinned to GPU VRAM successfully.")
        except Exception as e:
            logger.warning(f"Could not warm LLM '{config.ollama.llm_model}': {e}")

        # 2. Warm Router on CPU
        try:
            logger.info(f"Warming Router '{config.ollama.router_model}' on CPU (num_gpu={config.ollama.router_num_gpu})...")
            await self.generate(
                prompt="ping",
                model=config.ollama.router_model,
                options={"max_tokens": 1, "num_gpu": config.ollama.router_num_gpu},
                keep_alive=config.ollama.router_keep_alive,
                think=False,
            )
            status["router"] = True
            logger.info(f"Router '{config.ollama.router_model}' pinned to CPU successfully.")
        except Exception as e:
            logger.warning(f"Could not warm Router '{config.ollama.router_model}': {e}")

        # 3. Warm Embedding on CPU
        try:
            logger.info(f"Warming Embedding '{config.ollama.embedding_model}' on CPU (num_gpu={config.ollama.embedding_num_gpu})...")
            await self.get_embedding(
                "warmup query",
                model=config.ollama.embedding_model,
                num_gpu=config.ollama.embedding_num_gpu,
                keep_alive=config.ollama.embedding_keep_alive,
            )
            status["embedding"] = True
            logger.info(f"Embedding '{config.ollama.embedding_model}' pinned to CPU successfully.")
        except Exception as e:
            logger.warning(f"Could not warm Embedding '{config.ollama.embedding_model}': {e}")

        return status
