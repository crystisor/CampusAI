import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.config import config
from src.core.ollama_client import OllamaClient
from src.ingestion.ocr_engine import GLMOCREngine
from src.ingestion.pdf_processor import PDFPageData, PDFProcessor
from src.ingestion.pipeline import IngestionPipeline


def test_ocr_selection(monkeypatch):
    engine = GLMOCREngine()
    monkeypatch.setattr(config.ingestion, "ocr_min_text_chars", 80)
    monkeypatch.setattr(config.ingestion, "ocr_formula_pages", False)
    assert engine.ocr_reason(" \n")
    assert engine.ocr_reason("Short caption")
    text = "A normal course explanation. " * 8
    assert engine.ocr_reason(text) is None
    assert engine.ocr_reason(text + "\nx^2 + y^2 = 25") is None
    monkeypatch.setattr(config.ingestion, "ocr_formula_pages", True)
    assert engine.ocr_reason(text + "\nx^2 + y^2 = 25")
    assert engine.ocr_reason(text + "\n2026 12345") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {"message": {"content": "Recognized text"}},
    {"message": {"content": ""}},
    {"message": {"content": "Partial"}, "done_reason": "length"},
])
async def test_vision_api_contract(tmp_path, response):
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32)).save(image)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = MagicMock()
        post.return_value.json.return_value = response
        client = OllamaClient()
        if response["message"]["content"] == "Recognized text":
            assert await client.recognize_image(image, "glm-ocr", "1m") == "Recognized text"
        else:
            with pytest.raises(ValueError):
                await client.recognize_image(image, "glm-ocr", "1m")
        payload = post.call_args.kwargs["json"]
        assert post.call_args.args[0].endswith("/api/chat")
        assert payload["model"] == "glm-ocr"
        assert payload["keep_alive"] == "1m"
        assert base64.b64decode(payload["messages"][0]["images"][0]) == image.read_bytes()


@pytest.mark.asyncio
async def test_synthetic_preview_is_never_sent_to_ocr(tmp_path):
    image = tmp_path / "preview.png"
    Image.new("RGB", (32, 32)).save(image)
    client = MagicMock()
    client.recognize_image = AsyncMock()
    with pytest.raises(ValueError, match="real PDF page render"):
        await GLMOCREngine(ollama_client=client).recognize_page(PDFPageData(1, "", image))
    client.recognize_image.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure,scan_text", [(False, ""), (True, "short"), (True, "")])
async def test_hybrid_ingestion(tmp_path, monkeypatch, failure, scan_text):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    monkeypatch.setattr(config.ingestion, "ocr_min_text_chars", 80)
    monkeypatch.setattr(config.ingestion, "ocr_formula_pages", False)
    image = tmp_path / "scan.png"
    Image.new("RGB", (32, 32)).save(image)
    original = "Extracted course text about differential equations. " * 4
    processor = MagicMock()
    processor.extract_pages.return_value = [
        PDFPageData(1, original), PDFPageData(2, scan_text, image, True),
    ]
    client = MagicMock()
    recognized = "Recognized scanned content with equation $x^2$."
    client.recognize_image = AsyncMock(return_value=recognized)
    if failure:
        client.recognize_image.side_effect = RuntimeError("Ollama unavailable")
    client.get_embeddings = AsyncMock(side_effect=lambda texts, **kwargs: [[0.1] * 1024 for _ in texts])
    qdrant = MagicMock()
    qdrant.upsert_chunks = AsyncMock()
    pipeline = IngestionPipeline(pdf_processor=processor, ollama_client=client, qdrant_manager=qdrant)
    if failure and not scan_text:
        with pytest.raises(RuntimeError, match="ingestion stopped"):
            await pipeline.ingest_pdf(tmp_path / "course.pdf", "subject")
        qdrant.upsert_chunks.assert_not_called()
        client.get_embeddings.assert_not_called()
        return
    result = await pipeline.ingest_pdf(tmp_path / "course.pdf", "subject")
    client.recognize_image.assert_awaited_once()
    content = (tmp_path / "subject" / "course.md").read_text(encoding="utf-8")
    assert original in content
    assert (scan_text if failure else recognized) in content
    assert result["ocr_pages"] == ([] if failure else [2])
    assert bool(result["warnings"]) == failure
    qdrant.upsert_chunks.assert_awaited_once()


@pytest.mark.parametrize("rendered", [True, False])
def test_pdf_render_provenance(tmp_path, rendered):
    pdf = tmp_path / "input.pdf"
    pdf.touch()
    with patch("pypdf.PdfReader") as reader, patch("pdf2image.convert_from_path") as render:
        reader.return_value.pages = [MagicMock()]
        reader.return_value.pages[0].extract_text.return_value = "Native text"
        if rendered:
            render.return_value = [Image.new("RGB", (32, 32))]
        else:
            render.side_effect = RuntimeError("Poppler missing")
        page = PDFProcessor().extract_pages(pdf, tmp_path / "images")[0]
        assert page.text == "Native text"
        assert page.image_is_rendered is rendered
        assert page.image_path.exists()
