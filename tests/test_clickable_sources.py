from io import BytesIO
from unittest.mock import AsyncMock, Mock, call

import discord
import pytest

from src.bot.cogs import study_chat
from src.config import config


def course(document, page):
    return {"source": "course_material", "metadata": {
        "document_name": document, "page_number": page,
    }}


@pytest.mark.asyncio
async def test_exact_pages_are_published_once_and_linked(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    pdf = tmp_path / "software" / "raw" / "Paper_Unit_2.3_Extreme_Programming.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.touch()
    files = [discord.File(BytesIO(b"png"), filename=f"page_{n}.png") for n in (16, 14)]
    render = AsyncMock(side_effect=files)
    monkeypatch.setattr(study_chat, "render_formula_page", render)
    urls = [f"https://discord.com/channels/1/2/{n}" for n in (3, 4)]
    message = Mock(guild=Mock(me=None, filesize_limit=1000))
    message.channel.send = AsyncMock(side_effect=[Mock(jump_url=url) for url in urls])
    contexts = [course(pdf.stem, n) for n in (16, 14, 16)]

    links = await study_chat.publish_source_pages(message, contexts, "software")
    references = study_chat.format_references(contexts, links)

    assert render.await_args_list == [call((pdf, 16)), call((pdf, 14))]
    assert f"Slides/pages [16]({urls[0]}), [14]({urls[1]})" in references
    assert all(file.fp.closed for file in files)
    assert message.channel.send.await_count == 2
    parts = await study_chat.prepare_answer(references, None)
    assert all(f"[{n}]({url})" in "".join(part.text for part in parts) for n, url in zip((16, 14), urls))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["permission", "missing", "oversized", "render", "upload"])
async def test_unavailable_pages_preserve_visible_citations(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    pdf = tmp_path / "subject" / "raw" / "Lecture.pdf"
    pdf.parent.mkdir(parents=True)
    if failure != "missing":
        pdf.touch()
    file = discord.File(BytesIO(b"png"), filename="page.png")
    render = AsyncMock(return_value=file)
    if failure == "render":
        render.side_effect = ValueError("Invalid page")
    monkeypatch.setattr(study_chat, "render_formula_page", render)
    message = Mock(guild=Mock(me=Mock(), filesize_limit=1 if failure == "oversized" else 1000))
    message.channel.permissions_for.return_value.attach_files = failure != "permission"
    message.channel.send = AsyncMock(side_effect=OSError("Upload failed") if failure == "upload" else None)
    contexts = [course("Lecture", 16)]

    links = await study_chat.publish_source_pages(message, contexts, "subject")

    assert links == {}
    assert "Course: Lecture — Slide/page 16 (preview unavailable)" in study_chat.format_references(contexts, links)
    if failure in ("oversized", "upload"):
        assert file.fp.closed
    file.close()
    file.fp.close()


@pytest.mark.asyncio
async def test_invalid_sources_never_render_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    render = AsyncMock()
    monkeypatch.setattr(study_chat, "render_formula_page", render)
    outside = tmp_path / "outside.pdf"
    outside.touch()
    message = Mock(guild=Mock(me=None))
    contexts = [course("../../outside", 1), course("Lecture", 0), course("Lecture", True),
                course("Lecture", "16"), course("Lecture", None)]
    assert await study_chat.publish_source_pages(message, contexts, "subject") == {}
    assert await study_chat.publish_source_pages(message, [course("outside", 1)], "../escape") == {}
    render.assert_not_awaited()


def test_web_only_and_direct_references_unchanged():
    assert study_chat.format_references([], {}) == ""
    assert study_chat.format_references([
        {"source": "web_search", "metadata": {"url": "https://example.org"}},
    ], {}) == "\n\n**Sources:**\n- Web: https://example.org"
