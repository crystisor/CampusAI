"""Explicit TeX rendering convenience command."""
import math

import discord
from discord import app_commands
from discord.ext import commands

from src.bot.math_messages import Part, send_interaction_parts
from src.config import config


class LatexCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="latex", description="Render a mathematical TeX expression")
    @app_commands.guild_only()
    @app_commands.describe(code="Mathematical TeX", scale="Image scale", theme="Image colors")
    @app_commands.choices(theme=[app_commands.Choice(name="Light", value="light"), app_commands.Choice(name="Dark", value="dark")])
    async def latex_cmd(self, interaction: discord.Interaction, code: str,
                        scale: app_commands.Range[float, 0.5, 3.0] | None = None, theme: app_commands.Choice[str] | None = None):
        await interaction.response.defer(thinking=True)
        code = code.strip()
        for opening, closing in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")):
            if code.startswith(opening) and code.endswith(closing) and len(code) > len(opening) + len(closing):
                code = code[len(opening):-len(closing)]
                break
        code = code.strip()
        chosen_scale = config.latex.scale if scale is None else scale
        chosen_theme = config.latex.theme if theme is None else theme.value
        if not code or len(code) > config.latex.max_expression_chars or not math.isfinite(chosen_scale) or not 0.5 <= chosen_scale <= 3:
            await send_interaction_parts(interaction, [Part(f"Enter up to {config.latex.max_expression_chars:,} TeX characters and a scale from 0.5 to 3.")])
            return
        renderer = self.bot.latex_renderer
        result = (await renderer.render_many([code], scale=chosen_scale, theme=chosen_theme))[0]
        if not result.png:
            await send_interaction_parts(interaction, [Part(f"Could not render this expression ({result.error}).")])
            return
        await send_interaction_parts(interaction, [Part("**Equation 1**", result.png, "equation_1.png", code)])


async def setup(bot: commands.Bot):
    await bot.add_cog(LatexCog(bot))
