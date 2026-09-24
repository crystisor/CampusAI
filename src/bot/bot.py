import os
import asyncio
import logging
import discord
from discord.ext import commands

from src.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CampusAI.Bot")

class StudyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True

        super().__init__(
            command_prefix="!",
            intents=intents,
            help_command=None
        )

    async def setup_hook(self):
        # Load cogs
        cogs = [
            "src.bot.cogs.study_chat",
            "src.bot.cogs.admin",
        ]
        for cog in cogs:
            try:
                await self.load_extension(cog)
                logger.info(f"Loaded extension: {cog}")
            except Exception as e:
                logger.error(f"Failed to load extension {cog}: {e}", exc_info=True)

        # Sync application slash commands
        logger.info("Syncing slash commands...")
        try:
            synced = await self.tree.sync()
            logger.info(f"Successfully synced {len(synced)} slash commands.")
        except Exception as e:
            logger.error(f"Error syncing slash commands: {e}")

    async def on_ready(self):
        logger.info(f"Bot logged in as {self.user} (ID: {self.user.id})")
        activity = discord.CustomActivity(name=config.discord.status_message)
        await self.change_presence(activity=activity)
        logger.info("CampusAI Study Bot is online and listening!")

async def main():
    token = config.discord.token
    if not token or token == "YOUR_DISCORD_BOT_TOKEN":
        logger.warning(
            "DISCORD_BOT_TOKEN is not set in .env. "
            "Please configure DISCORD_BOT_TOKEN to connect the bot to Discord."
        )
        return

    bot = StudyBot()
    async with bot:
        await bot.start(token)

if __name__ == "__main__":
    asyncio.run(main())
