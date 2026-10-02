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

logger = logging.getLogger(__name__)


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
                )

                raw_answer = result.get("answer", "I could not generate an explanation at this time.")
                answer = clean_llm_response(raw_answer)
                decision = result.get("decision", "DIRECT")
                top_contexts = result.get("top_contexts", [])
                formula_page = source_formula_page(user_query, top_contexts, subject_id)
                formula_file = None
                if formula_page:
                    try:
                        formula_file = await render_formula_page(formula_page)
                    except Exception as exc:
                        logger.warning("Could not render source formula page: %s", exc)

                # Format response for Discord
                # Discord messages have a 2000 character limit per message
                # If longer, split into sequential messages
                reply_text = f"**[{subject_name}]** `Intent: {decision}`\n\n{answer}"
                if formula_file:
                    reply_text += "\n\n**Formula as printed in the PDF:** see the attached source page."

                # Append references if available
                if top_contexts:
                    sources_preview = "\n\n📚 **References Consulted:**\n"
                    for idx, ctx in enumerate(top_contexts[:3], 1):
                        source = ctx.get("source", "course")
                        meta = ctx.get("metadata", {})
                        doc_name = meta.get("document_name", "Material")
                        page = meta.get("page_number", "?")
                        if source == "course_material":
                            sources_preview += f"• `{doc_name}` (Page {page})\n"
                        else:
                            sources_preview += f"• Web: {ctx.get('metadata', {}).get('url', 'Search result')}\n"

                    # If fits in message limit, attach directly
                    if len(reply_text) + len(sources_preview) < 1950:
                        reply_text += sources_preview

                # Send message chunked if > 2000 chars
                if len(reply_text) <= 2000:
                    if formula_file:
                        await message.reply(reply_text, file=formula_file)
                    else:
                        await message.reply(reply_text)
                else:
                    chunks = [reply_text[i:i+1950] for i in range(0, len(reply_text), 1950)]
                    for chunk in chunks:
                        await message.channel.send(chunk)
                    if formula_file:
                        await message.channel.send(
                            "**Formula as printed in the PDF:**",
                            file=formula_file,
                        )

            except Exception as e:
                logger.error(f"Error handling message in #{message.channel.name}: {e}", exc_info=True)
                await message.reply(f"⚠️ An error occurred while processing your study query: `{str(e)}`")

async def setup(bot: commands.Bot):
    await bot.add_cog(StudyChatCog(bot))
