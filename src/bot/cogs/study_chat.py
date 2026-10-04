import logging
import re
import asyncio
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

logger = logging.getLogger(__name__)


def format_references(contexts: list[dict]) -> str:
    """List every source passed to answer generation, grouping repeated PDF pages."""
    course_pages: dict[str, list[str]] = {}
    web_sources: list[str] = []
    for context in contexts:
        metadata = context.get("metadata") or {}
        if context.get("source") == "course_material":
            document = str(metadata.get("document_name") or "Unknown course material")
            number = metadata.get("course_number")
            label = f"Course {number} ({document})" if number is not None else f"Course: {document}"
            page = metadata.get("page_number")
            page_label = str(page) if page is not None else "unknown"
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

class StudyChatCog(commands.Cog, name="StudyChat"):
    """
    Handles student questions in subject-bound channels.
    Routes queries through RAGPipeline (Arch-Router -> RAG/Web -> bge-reranker -> Spark-X2.5-4b-Q8_0).
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
            try:
                subject_name = subject_id.replace("_", " ").title()
                result = await self.pipeline.process_query(
                    query=user_query,
                    subject_id=subject_id,
                    subject_name=subject_name,
                    render_math=self.bot.latex_renderer.auto_available,
                )

                raw_answer = result.get("answer", "I could not generate an explanation at this time.")
                answer = clean_llm_response(raw_answer)
                decision = result.get("decision", "DIRECT")
                top_contexts = result.get("top_contexts", [])
                formula_page = source_formula_page(user_query, top_contexts, subject_id)
                parts = await prepare_answer(
                    answer, self.bot.latex_renderer,
                    header=f"**[{subject_name}]** `Intent: {decision}`\n\n",
                    footer=format_references(top_contexts),
                )
                if not await send_study_parts(message, parts):
                    return
                if formula_page:
                    formula_file = None
                    try:
                        can_attach = message.guild.me is None or message.channel.permissions_for(message.guild.me).attach_files
                        if not can_attach:
                            return
                        formula_file = await render_formula_page(formula_page)
                        if formula_file.fp.seek(0, 2) <= message.guild.filesize_limit:
                            formula_file.fp.seek(0)
                            await message.channel.send(
                                f"**Source PDF page:** {formula_page[0].name}, page {formula_page[1]}",
                                file=formula_file, allowed_mentions=MENTIONS,
                            )
                    except Exception as exc:
                        logger.warning("Could not deliver source PDF page in channel %s (%s)", message.channel.id, type(exc).__name__)
                    finally:
                        if formula_file:
                            formula_file.close()
                            formula_file.fp.close()

            except Exception as e:
                logger.error(f"Error handling message in #{message.channel.name}: {e}", exc_info=True)
                await send_study_parts(message, [Part("An error occurred while processing your study query.")])

async def setup(bot: commands.Bot):
    await bot.add_cog(StudyChatCog(bot))
