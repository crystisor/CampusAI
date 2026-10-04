"""Bounded local MathJax subprocess bridge."""
import asyncio
import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import json
import logging

from PIL import Image

from src.config import BASE_DIR, LatexSettings

logger = logging.getLogger(__name__)
HELPER = BASE_DIR / "tools" / "latex-renderer" / "render.mjs"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class RenderResult:
    png: bytes | None = None
    width: int = 0
    height: int = 0
    error: str | None = None


async def _read_limited(stream, limit: int) -> bytes:
    chunks = []
    size = 0
    while chunk := await stream.read(min(65536, limit + 1 - size)):
        size += len(chunk)
        if size > limit:
            raise ValueError("process output exceeded limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _discard(stream) -> None:
    while await stream.read(65536):
        pass


class LatexRenderer:
    def __init__(self, settings: LatexSettings):
        self.settings = settings
        self.available = False
        self._lock = asyncio.Lock()
        self._children: set[asyncio.subprocess.Process] = set()
        self._closed = False

    @property
    def auto_available(self) -> bool:
        return self.settings.enabled and self.settings.auto_render and self.available and not self._closed

    async def probe(self) -> bool:
        if not self.settings.enabled or self._closed:
            return False
        result = await self.render_many(["x"], scale=self.settings.scale, theme=self.settings.theme, probing=True)
        if result[0].error == "busy":
            return self.available
        self.available = bool(result and result[0].png)
        if not self.available:
            logger.warning("LaTeX renderer unavailable; install Node dependencies with npm ci --prefix tools/latex-renderer")
        return self.available

    async def render_many(self, expressions: list[str], *, scale: float, theme: str, probing: bool = False) -> list[RenderResult]:
        if not expressions:
            return []
        if not self.settings.enabled or self._closed or (not probing and not self.available):
            return [RenderResult(error="unavailable") for _ in expressions]
        if self._lock.locked():
            return [RenderResult(error="busy") for _ in expressions]
        if len(expressions) > 4 or not 0.5 <= scale <= 3 or theme not in ("light", "dark"):
            return [RenderResult(error="invalid_input") for _ in expressions]
        valid = [isinstance(e, str) and bool(e.strip()) and len(e) <= self.settings.max_expression_chars for e in expressions]
        if not all(valid):
            rendered = iter(await self.render_many([e for e, ok in zip(expressions, valid) if ok], scale=scale, theme=theme, probing=probing))
            return [next(rendered) if ok else RenderResult(error="invalid_input") for ok in valid]
        payload = json.dumps({"version": 1, "expressions": expressions, "scale": scale, "theme": theme}, ensure_ascii=False).encode()
        if len(payload) > 65536:
            return [RenderResult(error="invalid_input") for _ in expressions]
        async with self._lock:
            process = None
            readers = []
            try:
                async with asyncio.timeout(self.settings.timeout_seconds):
                    process = await asyncio.create_subprocess_exec(
                        self.settings.node_executable, str(HELPER),
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    self._children.add(process)
                    if self._closed:
                        return [RenderResult(error="unavailable") for _ in expressions]
                    readers = [
                        asyncio.create_task(_read_limited(process.stdout, 6 * 1024 * 1024)),
                        asyncio.create_task(_read_limited(process.stderr, 16 * 1024)),
                    ]
                    process.stdin.write(payload)
                    await process.stdin.drain()
                    process.stdin.close()
                    stdout, stderr, _ = await asyncio.gather(
                        *readers, process.wait(),
                    )
                if process.returncode != 0:
                    logger.warning("LaTeX renderer exited with code %s", process.returncode)
                    raise OSError("renderer failed")
                document = json.loads(stdout)
                if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] != 1 or not isinstance(document.get("results"), list) or len(document["results"]) != len(expressions):
                    raise ValueError("invalid renderer response")
                return [self._decode(item) for item in document["results"]]
            except asyncio.CancelledError:
                raise
            except (OSError, ValueError, asyncio.TimeoutError, NotImplementedError) as exc:
                self.available = False
                logger.warning("LaTeX renderer process unavailable: %s", type(exc).__name__)
                category = ("timeout" if isinstance(exc, asyncio.TimeoutError) else
                            "protocol_error" if isinstance(exc, ValueError) else "unavailable")
                return [RenderResult(error=category) for _ in expressions]
            finally:
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                if process is not None:
                    if process.stdin:
                        process.stdin.close()
                    if process.returncode is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                    await asyncio.gather(_discard(process.stdout), _discard(process.stderr), process.wait())
                    self._children.discard(process)

    def _decode(self, item: dict) -> RenderResult:
        if not isinstance(item, dict):
            raise ValueError("invalid result")
        if item.get("ok") is False:
            category = item.get("error")
            if not isinstance(category, str) or category not in {"invalid_input", "syntax", "unsupported", "too_large"}:
                raise ValueError("unknown result error")
            return RenderResult(error=category)
        if item.get("ok") is not True:
            raise ValueError("invalid result status")
        try:
            data = base64.b64decode(item["png_base64"], validate=True)
        except (KeyError, TypeError, binascii.Error) as exc:
            raise ValueError("invalid PNG data") from exc
        width, height = item.get("width"), item.get("height")
        if (not data.startswith(PNG_MAGIC) or len(data) > 1048576 or
                type(width) is not int or type(height) is not int or width < 1 or height < 1 or
                width > 2048 or height > 1024 or width * height > 2097152):
            raise ValueError("invalid PNG size")
        try:
            with Image.open(BytesIO(data)) as image:
                image.verify()
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG" or image.size != (width, height):
                    raise ValueError("PNG dimensions differ")
                image.load()
        except Exception as exc:
            raise ValueError("invalid PNG") from exc
        if (len(data) > self.settings.max_png_bytes or width > self.settings.max_width or
                height > self.settings.max_height or width * height > self.settings.max_pixels):
            return RenderResult(error="too_large")
        return RenderResult(data, width, height)

    async def close(self) -> None:
        self._closed = True
        for process in tuple(self._children):
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        await asyncio.gather(*(p.wait() for p in tuple(self._children)), return_exceptions=True)
        # Wait for in-flight creation and its finally block to release admission.
        async with self._lock:
            pass
        self.available = False
