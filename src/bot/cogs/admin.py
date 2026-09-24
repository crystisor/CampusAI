import logging
import discord
from discord import app_commands
from discord.ext import commands
from typing import Optional

from src.bot.channel_manager import channel_manager
from src.rag.qdrant_manager import QdrantManager
from src.core.ollama_client import OllamaClient
from src.rag.pipeline import RAGPipeline
from src.config import config

logger = logging.getLogger(__name__)

class AdminCog(commands.Cog, name="Admin"):
    """
    Administration and diagnostics commands for subject channel bindings and system state.
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.qdrant = QdrantManager()
        self.ollama = OllamaClient()
        self.pipeline = RAGPipeline(qdrant_manager=self.qdrant, ollama_client=self.ollama)

    @app_commands.command(name="bind", description="Bind this channel to a study subject ID (e.g. calculus_1)")
    @app_commands.describe(subject_id="Unique identifier of the subject")
    async def bind_channel_cmd(self, interaction: discord.Interaction, subject_id: str):
        sub_id = "".join(c if c.isalnum() else "_" for c in subject_id.lower()).strip("_")
        channel_manager.bind_channel(interaction.channel_id, sub_id, channel_name=f"#{interaction.channel.name}")
        await interaction.response.send_message(
            f"✅ Successfully bound channel **#{interaction.channel.name}** to subject **`{sub_id}`**.\n"
            f"All questions asked in this channel will now be answered using `{sub_id}` course materials!",
            ephemeral=False
        )

    @app_commands.command(name="unbind", description="Unbind this channel from its study subject")
    async def unbind_channel_cmd(self, interaction: discord.Interaction):
        removed = channel_manager.unbind_channel(interaction.channel_id)
        if removed:
            await interaction.response.send_message(
                f"ℹ️ Channel **#{interaction.channel.name}** has been unbound.",
                ephemeral=False
            )
        else:
            await interaction.response.send_message(
                f"ℹ️ Channel **#{interaction.channel.name}** was not bound to any subject.",
                ephemeral=True
            )

    @app_commands.command(name="status", description="Inspect AI model stack, Qdrant database, and channel bindings")
    async def status_cmd(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=False)

        bound_sub = channel_manager.get_subject_for_channel(interaction.channel_id)
        ollama_ok = await self.ollama.is_available()
        qdrant_ok = await self.qdrant.is_available()

        embed = discord.Embed(
            title="CampusAI System Health & Diagnostics",
            color=discord.Color.blue() if (ollama_ok and qdrant_ok) else discord.Color.orange()
        )

        embed.add_field(
            name="Current Channel",
            value=f"#{interaction.channel.name}\nBound Subject: **`{bound_sub or 'None'}`**",
            inline=True
        )

        ollama_status = "🟢 Online" if ollama_ok else "🔴 Offline"
        qdrant_status = "🟢 Online" if qdrant_ok else "🔴 Offline"

        embed.add_field(
            name="Core Services",
            value=f"• Ollama (`Spark-X2.5-4B`): {ollama_status}\n• Qdrant Vector DB: {qdrant_status}",
            inline=True
        )

        if bound_sub:
            stats = await self.qdrant.get_subject_stats(bound_sub)
            embed.add_field(
                name="Vector Index Stats",
                value=f"Collection: `{stats.get('collection_name')}`\nVectors: `{stats.get('points_count', 0)}`",
                inline=False
            )

        embed.set_footer(text="CampusAI • RTX 2060 SUPER 8GB Optimized")
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="subjects", description="List all registered college study subjects")
    async def subjects_cmd(self, interaction: discord.Interaction):
        subjects = await self.qdrant.list_all_subjects()
        all_bindings = channel_manager.get_all_bindings()

        if not subjects:
            await interaction.response.send_message("No subjects have been indexed in Qdrant yet.", ephemeral=True)
            return

        lines = []
        for s in subjects:
            channels = channel_manager.get_channels_for_subject(s)
            ch_str = ", ".join(channels) if channels else "No channels bound"
            lines.append(f"• **`{s}`** — {ch_str}")

        desc = "\n".join(lines)
        embed = discord.Embed(
            title="Registered Study Subjects",
            description=desc,
            color=discord.Color.green()
        )
        await interaction.response.send_message(embed=embed)

    @app_commands.command(name="ask", description="Ask CampusAI a question directly")
    @app_commands.describe(question="The question to ask", subject_id="Optional subject ID override")
    async def ask_cmd(self, interaction: discord.Interaction, question: str, subject_id: Optional[str] = None):
        target_sub = subject_id or channel_manager.get_subject_for_channel(interaction.channel_id) or "general"
        await interaction.response.defer(thinking=True)

        try:
            res = await self.pipeline.process_query(
                query=question,
                subject_id=target_sub,
                subject_name=target_sub.replace("_", " ").title()
            )
            answer = res.get("answer", "")
            decision = res.get("decision", "DIRECT")

            reply = f"**[{target_sub.upper()}]** `Intent: {decision}`\n\n{answer}"
            if len(reply) <= 2000:
                await interaction.followup.send(reply)
            else:
                chunks = [reply[i:i+1950] for i in range(0, len(reply), 1950)]
                await interaction.followup.send(chunks[0])
                for chunk in chunks[1:]:
                    await interaction.channel.send(chunk)
        except Exception as e:
            await interaction.followup.send(f"⚠️ Error answering question: `{e}`")

async def setup(bot: commands.Bot):
    await bot.add_cog(AdminCog(bot))
