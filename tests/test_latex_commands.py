from unittest.mock import AsyncMock, Mock
from io import BytesIO
from pathlib import Path

import pytest
import discord
import src.bot.cogs.study_chat as study_chat

from src.bot.cogs.latex import LatexCog
from src.bot.bot import StudyBot
from src.bot.latex_renderer import RenderResult
from src.bot.cogs.admin import AdminCog
from src.bot.cogs.study_chat import StudyChatCog
from src.config import LatexSettings


@pytest.mark.asyncio
async def test_latex_command_defers_and_uses_original_response():
    bot = Mock(latex_renderer=Mock(render_many=AsyncMock(return_value=[RenderResult(png=b"png", width=10, height=10)])))
    cog = LatexCog(bot)
    interaction = Mock(response=Mock(defer=AsyncMock()), edit_original_response=AsyncMock(), guild=None)
    async def render(*args, **kwargs):
        interaction.response.defer.assert_awaited_once()
        return [RenderResult(png=b"png", width=10, height=10)]
    bot.latex_renderer.render_many.side_effect = render
    await cog.latex_cmd.callback(cog, interaction, "  $$x=1$$  ")
    interaction.response.defer.assert_awaited_once()
    assert bot.latex_renderer.render_many.call_args.args == (["x=1"],)
    assert "attachments" in interaction.edit_original_response.call_args.kwargs


@pytest.mark.asyncio
async def test_all_existing_commands_remain_registered(monkeypatch):
    monkeypatch.setattr("src.rag.pipeline.RAGPipeline", Mock())
    monkeypatch.setattr("src.rag.qdrant_manager.QdrantManager", Mock())
    monkeypatch.setattr("src.core.ollama_client.OllamaClient", Mock())
    bot = StudyBot()
    try:
        for module in ("src.bot.cogs.study_chat", "src.bot.cogs.admin", "src.bot.cogs.latex"):
            await bot.load_extension(module)
        assert {command.name for command in bot.tree.get_commands()} >= {
            "ask", "bind", "unbind", "status", "subjects", "latex",
        }
    finally:
        await bot.close()


@pytest.mark.asyncio
async def test_ask_automatically_renders_and_preserves_sources():
    renderer = Mock(auto_available=True, settings=LatexSettings(), render_many=AsyncMock(return_value=[RenderResult(png=b"png")]))
    cog = object.__new__(AdminCog)
    cog.bot = Mock(latex_renderer=renderer)
    cog.pipeline = Mock(process_query=AsyncMock(return_value={
        "answer": "Solve using $$x=2$$ then check.", "decision": "DIRECT",
        "top_contexts": [{"source": "web_search", "metadata": {"url": "https://example.org"}}],
    }))
    interaction = Mock(guild=None, response=Mock(defer=AsyncMock()), edit_original_response=AsyncMock(), followup=Mock(send=AsyncMock()))
    await cog.ask_cmd.callback(cog, interaction, "How do I solve this?", "math")
    assert cog.pipeline.process_query.call_args.kwargs["render_math"] is True
    renderer.render_many.assert_awaited_once()
    assert interaction.edit_original_response.call_args.kwargs["content"].startswith("**[MATH]**")
    calls = interaction.followup.send.call_args_list
    assert calls[0].kwargs["file"].filename == "equation_1.png"
    assert "https://example.org" in calls[-1].args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("source_page", [False, True, "rejected"])
async def test_study_reply_generated_math_and_source_page_coexist(monkeypatch, source_page):
    renderer = Mock(auto_available=True, settings=LatexSettings(), render_many=AsyncMock(return_value=[RenderResult(png=b"png")]))
    cog = object.__new__(StudyChatCog)
    cog.bot = Mock(latex_renderer=renderer)
    cog.pipeline = Mock(process_query=AsyncMock(return_value={
        "answer": "The solution is $$x=2$$.", "decision": "RAG",
        "top_contexts": [{"source": "course_material", "metadata": {"document_name": "Lecture", "course_number": 3, "page_number": 7}}],
    }))
    monkeypatch.setattr(study_chat.channel_manager, "get_subject_for_channel", lambda _: "math")
    monkeypatch.setattr(study_chat, "source_formula_page", lambda *args: (Path("Lecture.pdf"), 7) if source_page else None)
    pdf = discord.File(BytesIO(b"pdf-image"), filename="source_page.png")
    page_render = AsyncMock(return_value=pdf)
    monkeypatch.setattr(study_chat, "render_formula_page", page_render)
    channel = Mock(id=123, typing=Mock(return_value=AsyncMock()), send=AsyncMock())
    if source_page == "rejected":
        async def reject_pdf(*args, **kwargs):
            if kwargs.get("file") and kwargs["file"].filename == "source_page.png":
                raise discord.HTTPException(Mock(status=413, reason="too large"), {"code": 40005, "message": "too large"})
        channel.send.side_effect = reject_pdf
    message = Mock(author=Mock(bot=False), guild=Mock(me=None, filesize_limit=1024), channel=channel, content="How do I solve a quadratic equation?", reply=AsyncMock())
    await cog.on_message(message)
    assert cog.pipeline.process_query.call_args.kwargs["render_math"] is True
    message.reply.assert_awaited_once()
    files = [call.kwargs["file"].filename for call in channel.send.call_args_list if "file" in call.kwargs]
    assert files == (["equation_1.png", "source_page.png"] if source_page else ["equation_1.png"])
    assert any("Course 3 (Lecture)" in call.args[0] for call in channel.send.call_args_list)
    if source_page:
        assert "Source PDF page" in channel.send.call_args.args[0]
        assert pdf.fp.closed
    else:
        pdf.close()
        page_render.assert_not_awaited()
