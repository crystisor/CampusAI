import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from src.config import config
from src.core.ollama_client import EmptyImageResultError, OllamaClient
from src.ingestion.pdf_processor import PDFPageData, PDFProcessor
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vision_engine import PageVisionEngine


@pytest.mark.parametrize("operators,expected", [
    (b"", False),
    (b"10 10 m 100 100 l S", True),  # Vector chart, no embedded bitmap.
    (b"10 10 100 100 re f", True),
])
def test_pdf_visual_evidence(tmp_path, operators, expected):
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    stream = DecodedStreamObject()
    stream.set_data(operators)
    page[NameObject("/Contents")] = writer._add_object(stream)
    pdf = tmp_path / "visual.pdf"
    writer.write(pdf)
    with patch("pdf2image.convert_from_path", return_value=[Image.new("RGB", (200, 200))]):
        result = PDFProcessor().extract_pages(pdf, tmp_path / "images")[0]
    assert result.has_visual_content is expected


@pytest.mark.asyncio
async def test_vision_skips_text_and_synthetic_previews(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ingestion, "vision_enabled", True)
    client = MagicMock(recognize_image=AsyncMock())
    engine = PageVisionEngine(client)
    text_page = PDFPageData(1, "Long ordinary prose. " * 20)
    assert not engine.needs_vision(text_page)
    text_page.has_visual_content = True
    assert engine.needs_vision(text_page)
    with pytest.raises(ValueError, match="real PDF page render"):
        await engine.describe_page(text_page)
    client.recognize_image.assert_not_called()
    monkeypatch.setattr(config.ingestion, "vision_enabled", False)
    assert not engine.needs_vision(text_page)


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["Class B has twice as many students.", ""])
async def test_gemma_vision_request(tmp_path, output):
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32)).save(image)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = MagicMock()
        post.return_value.json.return_value = {"message": {
            "content": json.dumps({"has_visual_content": bool(output), "description": output}), "thinking": "private",
        }}
        result = await PageVisionEngine(OllamaClient()).describe_page(PDFPageData(1, "", image, True))
        assert result == output
        payload = post.call_args.kwargs["json"]
        assert payload["model"] == config.ingestion.vision_model
        assert payload["think"] is False
        assert payload["format"]["required"] == ["has_visual_content", "description"]
        assert payload["options"]["num_ctx"] == config.ollama.llm_num_ctx
        assert payload["options"]["num_predict"] == config.ingestion.vision_max_tokens
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["images"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_visual_descriptions_are_indexed_with_page_text(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    monkeypatch.setattr(config.ingestion, "vision_enabled", True)
    monkeypatch.setattr(config.ingestion, "ocr_formula_pages", False)
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32)).save(image)
    prose = "The course explains changes in student attendance. " * 3
    processor = MagicMock()
    processor.extract_pages.return_value = [
        PDFPageData(1, prose, image, True),
        PDFPageData(2, prose, image, True, True),
        PDFPageData(3, "", image, True, True),
    ]
    description = "Class B has 40 students compared with 20 in Class A."

    async def recognize(path, model, **kwargs):
        if model == config.ingestion.ocr_model:
            return "Scanned course notes."
        if failure:
            raise RuntimeError("vision unavailable")
        return json.dumps({"has_visual_content": True, "description": description})

    client = MagicMock(recognize_image=AsyncMock(side_effect=recognize))
    client.get_embeddings = AsyncMock(side_effect=lambda texts, **kw: [[0.1]*1024 for _ in texts])
    qdrant = MagicMock(upsert_chunks=AsyncMock())
    result = await IngestionPipeline(
        pdf_processor=processor, ollama_client=client, qdrant_manager=qdrant,
    ).ingest_pdf(tmp_path / "course.pdf", "subject")
    calls = client.recognize_image.await_args_list
    assert [c.kwargs["model"] for c in calls] == [
        config.ingestion.vision_model, config.ingestion.ocr_model, config.ingestion.vision_model,
    ]
    assert result["ocr_pages"] == [3]
    assert result["vision_pages"] == ([] if failure else [2, 3])
    assert len(result["warnings"]) == (2 if failure else 0)
    chunks = qdrant.upsert_chunks.call_args.args[1]
    assert any(prose.strip() in chunk["text"] for chunk in chunks)
    assert any("Scanned course notes." in chunk["text"] for chunk in chunks)
    described = [chunk for chunk in chunks if description in chunk["text"]]
    assert bool(described) is not failure
    assert all(chunk["page_number"] in (2, 3) for chunk in described)


@pytest.mark.asyncio
@pytest.mark.parametrize("description", ["A photograph of a red circuit board.", ""])
async def test_image_without_text_uses_vision_or_stops(tmp_path, monkeypatch, description):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    monkeypatch.setattr(config.ingestion, "vision_enabled", True)
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32)).save(image)
    processor = MagicMock()
    processor.extract_pages.return_value = [PDFPageData(1, "", image, True, True)]
    client = MagicMock(recognize_image=AsyncMock(side_effect=[
        EmptyImageResultError("OCR returned no text"),
        json.dumps({"has_visual_content": bool(description), "description": description}),
    ]))
    client.get_embeddings = AsyncMock(side_effect=lambda texts, **kw: [[0.1]*1024 for _ in texts])
    sink = MagicMock(upsert_chunks=AsyncMock())
    pipeline = IngestionPipeline(pdf_processor=processor, ollama_client=client, qdrant_manager=sink)
    if description:
        result = await pipeline.ingest_pdf(tmp_path / "image.pdf", "subject")
        assert result["vision_pages"] == [1]
        assert description in sink.upsert_chunks.call_args.args[1][0]["text"]
    else:
        with pytest.raises(RuntimeError, match="ingestion stopped"):
            await pipeline.ingest_pdf(tmp_path / "image.pdf", "subject")
        sink.upsert_chunks.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["not JSON", '{"has_visual_content":"false","description":""}',
                                     '{"has_visual_content":true,"description":""}'])
async def test_invalid_visual_assessments_are_rejected(tmp_path, output):
    image = tmp_path / "page.png"
    Image.new("RGB", (32, 32)).save(image)
    client = MagicMock(recognize_image=AsyncMock(return_value=output))
    with pytest.raises(ValueError):
        await PageVisionEngine(client).describe_page(PDFPageData(1, "", image, True))
