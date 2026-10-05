import logging
import re
import asyncio
from time import monotonic
from io import BytesIO
from pathlib import Path
from pdf2image import convert_from_path
import discord
from discord.ext import commands
from typing import Optional

from src.bot.channel_manager import channel_manager
from src.rag.pipeline import RAGPipeline
from src.core.ollama_client import clean_llm_response
from src.config import config
from src.bot.math_messages import Part, prepare_answer, send_study_parts, MENTIONS
from src.bot.session_store import session_store

logger = logging.getLogger(__name__)


def format_references(contexts: list[dict], page_links: Optional[dict[tuple[str, str], str]] = None) -> str:
    """List every source passed to answer generation, grouping repeated PDF pages."""
    course_pages: dict[str, list[str]] = {}
    web_sources: list[str] = []
    for context in contexts:
        metadata = context.get("metadata") or {}
        if context.get("source") == "course_material":
            document = str(metadata.get("document_name") or "Unknown course material")
            number = metadata.get("course_number")
            display_document = discord.utils.escape_markdown(document)
            label = f"Course {number} ({display_document})" if number is not None else f"Course: {display_document}"
            page = metadata.get("page_number")
            page_label = str(page) if page is not None else "unknown"
            if page_links is not None:
                url = page_links.get((document, page_label))
                page_label = f"[{page_label}]({url})" if url else f"{page_label} (preview unavailable)"
            pages = course_pages.setdefault(label, [])
            if page_label not in pages:
                pages.append(page_label)
        elif context.get("source") == "web_search":
            url = str(metadata.get("url") or "Search result")
            if url not in web_sources:
                web_sources.append(url)

    if not course_pages and not web_sources:
        return ""
    lines = ["**Sources:**"]
    for label, pages in course_pages.items():
        kind = "Slide/page" if len(pages) == 1 else "Slides/pages"
        lines.append(f"- {label} — {kind} {', '.join(pages)}")
    lines.extend(f"- Web: {url}" for url in web_sources)
    return "\n\n" + "\n".join(lines)


def source_formula_page(query: str, contexts: list[dict], subject_id: str) -> Optional[tuple[Path, int]]:
    """Find the original PDF page for a request about a source formula."""
    if not re.search(r"\b(formula|equation|expression)\b", query, re.IGNORECASE):
        return None

    subject_dir = (config.ingestion.storage_dir / subject_id).resolve()
    raw_dir = subject_dir / "raw"
    for context in contexts:
        if context.get("source") != "course_material":
            continue
        metadata = context.get("metadata", {})
        doc_name = metadata.get("document_name")
        page_number = metadata.get("page_number")
        if not isinstance(doc_name, str) or not doc_name or not isinstance(page_number, int) or page_number < 1:
            continue
        pdf = (raw_dir / f"{doc_name}.pdf").resolve()
        if pdf.is_relative_to(raw_dir.resolve()) and pdf.is_file():
            return pdf, page_number
    return None


async def render_formula_page(source: tuple[Path, int]) -> discord.File:
    """Render the actual PDF page so its formula keeps the original typesetting."""
    pdf, page_number = source
    pages = await asyncio.to_thread(
        convert_from_path, str(pdf), dpi=config.ingestion.dpi,
        first_page=page_number, last_page=page_number,
    )
    if not pages:
        raise ValueError(f"Could not render page {page_number} of {pdf.name}")
    output = BytesIO()
    pages[0].save(output, format="PNG")
    output.seek(0)
    return discord.File(output, filename=f"{pdf.stem}_page_{page_number}.png")


async def publish_source_pages(message: discord.Message, contexts: list[dict], subject_id: str) -> dict[tuple[str, str], str]:
    """Publish exact PDF pages and retain durable, channel-scoped message links."""
    links: dict[tuple[str, str], str] = {}
    if message.guild.me is not None and not message.channel.permissions_for(message.guild.me).attach_files:
        return links
    root = config.ingestion.storage_dir.resolve()
    subject_dir = (root / subject_id).resolve()
    if not subject_dir.is_relative_to(root):
        return links
    raw_dir = (subject_dir / "raw").resolve()
    if not raw_dir.is_relative_to(subject_dir):
        return links
    seen = set()
    for context in contexts:
        metadata = context.get("metadata") or {}
        document, page = metadata.get("document_name"), metadata.get("page_number")
        if (context.get("source") != "course_material" or not isinstance(document, str)
                or not document or type(page) is not int or page < 1):
            continue
        key = (document, str(page))
        if key in seen:
            continue
        seen.add(key)
        pdf = (raw_dir / f"{document}.pdf").resolve()
        if not pdf.is_relative_to(raw_dir) or not pdf.is_file():
            continue
        file = None
        try:
            file = await render_formula_page((pdf, page))
            if file.fp.seek(0, 2) > message.guild.filesize_limit:
                continue
            file.fp.seek(0)
            source_message = await message.channel.send(
                f"**Course:** {discord.utils.escape_markdown(document)[:1500]} — **Slide/page {page}**",
                file=file, allowed_mentions=MENTIONS,
            )
            links[key] = source_message.jump_url
        except Exception:
            logger.warning("Could not publish source page %s of %s", page, document, exc_info=True)
        finally:
            if file:
                file.close()
                file.fp.close()
    return links

class StudyChatCog(commands.Cog, name="StudyChat"):
    """
    Handles student questions in subject-bound channels.
    Routes queries through RAGPipeline (Arch-Router -> RAG/Web -> bge-reranker -> gemma4_e2b_q8:latest).
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.pipeline = RAGPipeline()

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        # Ignore messages sent by bots or system
        if message.author.bot or not message.guild:
            return

        # Check if the channel is bound to a subject
        subject_id = channel_manager.get_subject_for_channel(message.channel.id)
        if not subject_id:
            return

        # Ignore slash commands or bot prefix commands if any
        if message.content.startswith("/"):
            return

        user_query = message.content.strip()
        if not user_query:
            return

        # Indicate typing while performing routing, retrieval, reranking, and generation
        async with message.channel.typing():
            stream_messages: list[discord.Message] = []
            session = None
            session_task = asyncio.current_task()
            try:
                subject_name = subject_id.replace("_", " ").title()
                session = await session_store.get(message.channel.id) if type(message.channel.id) is int else None
                if session:
                    session_store.register_task(message.channel.id, session_task)
                summary, turns = await session_store.context(session.id) if session else ("", [])
                if session:
                    await session_store.append(session.id, message.channel.id, message.author.display_name, "user", user_query,
                                               subject_id=subject_id, event_id=message.id)
                history = [{"role": turn["role"], "content": f"{turn['speaker']} (subject {turn['subject_id'] or 'unknown'}): {turn['content']}"} for turn in turns]
                header = f"**[{subject_name}]** `Generating answer...`\n\n"
                stream_message = await message.reply(header, allowed_mentions=MENTIONS)
                stream_messages.append(stream_message)
                streamed = ""
                last_edit = monotonic()

                async def show_token(token: str) -> None:
                    nonlocal stream_message, streamed, last_edit
                    # Keep each Discord message below its 2,000 character limit.
                    while token:
                        room = 1850 - len(streamed)
                        if room <= 0:
                            stream_message = await message.channel.send(
                                streamed, allowed_mentions=MENTIONS
                            )
                            stream_messages.append(stream_message)
                            streamed = ""
                            room = 1850
                        streamed += token[:room]
                        token = token[room:]
                    now = monotonic()
                    if now - last_edit >= 0.8:
                        await stream_message.edit(
                            content=(header + streamed) if stream_message is stream_messages[0] else streamed,
                            allowed_mentions=MENTIONS,
                        )
                        last_edit = now

                result = await self.pipeline.process_query(
                    query=user_query,
                    subject_id=subject_id,
                    subject_name=subject_name,
                    render_math=self.bot.latex_renderer.auto_available,
                    on_token=show_token,
                    conversation_summary=summary,
                    conversation_history=history,
                )

                raw_answer = result.get("answer", "I could not generate an explanation at this time.")
                if session and not await session_store.is_active(session.id, message.channel.id):
                    return
                answer = clean_llm_response(raw_answer)
                decision = result.get("decision", "DIRECT")
                top_contexts = result.get("top_contexts", [])
                parts = await prepare_answer(
                    answer, self.bot.latex_renderer,
                    header=f"**[{subject_name}]** `Intent: {decision}`\n\n",
                )
                # The live text is a preview. Replace it with the established
                # final delivery so equation rendering, citations and splitting remain intact.
                for streamed_message in stream_messages:
                    try:
                        await streamed_message.delete()
                    except discord.HTTPException:
                        logger.debug("Could not remove streamed preview message %s", streamed_message.id)
                if not await send_study_parts(message, parts):
                    return
                if session:
                    await session_store.append(session.id, message.channel.id, "CampusAI", "assistant", answer,
                                               subject_id=subject_id, source_metadata={"contexts": top_contexts})
                page_links = await publish_source_pages(message, top_contexts, subject_id)
                references = format_references(top_contexts, page_links)
                if references:
                    source_parts = await prepare_answer(references, None)
                    await send_study_parts(message, source_parts)

            except asyncio.CancelledError:
                for streamed_message in stream_messages:
                    try:
                        await streamed_message.delete()
                    except discord.HTTPException:
                        pass
                raise
            except Exception as e:
                logger.error(f"Error handling message in #{message.channel.name}: {e}", exc_info=True)
                for streamed_message in stream_messages:
                    try:
                        await streamed_message.delete()
                    except discord.HTTPException:
                        pass
                await send_study_parts(message, [Part("An error occurred while processing your study query.")])
            finally:
                if session:
                    session_store.unregister_task(message.channel.id, session_task)

async def setup(bot: commands.Bot):
    await bot.add_cog(StudyChatCog(bot))
