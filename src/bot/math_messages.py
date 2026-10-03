"""Turn marked math in an answer into ordered Discord text and image parts."""
from dataclasses import dataclass
from io import BytesIO
import logging
import re

import aiohttp
import discord

from src.bot.latex_renderer import LatexRenderer

MAX_TEXT = 1950
MENTIONS = discord.AllowedMentions.none()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Part:
    text: str
    png: bytes | None = None
    filename: str | None = None
    expression: str | None = None


def _escaped(value: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and value[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def segments(answer: str) -> list[tuple[str, str]]:
    """Scan math outside Markdown code, retaining all malformed text verbatim."""
    result: list[tuple[str, str]] = []
    plain = []
    i = 0
    line_start = True
    fence = None
    code_ticks = 0
    while i < len(answer):
        if line_start and not code_ticks:
            match = re.match(r" {0,3}(`{3,}|~{3,})", answer[i:])
            if match:
                marker = match.group(1)
                if fence is None:
                    fence = (marker[0], len(marker))
                elif marker[0] == fence[0] and len(marker) >= fence[1] and not answer[i + len(match.group(0)):].split("\n", 1)[0].strip():
                    fence = None
                plain.append(match.group(0))
                i += len(match.group(0))
                line_start = False
                continue
        ch = answer[i]
        if ch == "\n":
            line_start = True
            plain.append(ch)
            i += 1
            continue
        if fence:
            line_start = False
            plain.append(ch)
            i += 1
            continue
        if ch == "`" and not _escaped(answer, i):
            run = len(answer[i:]) - len(answer[i:].lstrip("`"))
            if code_ticks == 0:
                closing = re.search(r"(?<!`)`{" + str(run) + r"}(?!`)", answer[i + run:])
                if closing:
                    code_ticks = run
            elif run == code_ticks:
                code_ticks = 0
            plain.append("`" * run)
            i += run
            line_start = False
            continue
        if not code_ticks and not _escaped(answer, i):
            opener = next((pair for pair in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)")) if answer.startswith(pair[0], i)), None)
            if opener:
                start = i + len(opener[0])
                end = start
                while end < len(answer):
                    if answer.startswith(opener[1], end) and not _escaped(answer, end):
                        break
                    end += 1
                span = answer[start:end]
                if end < len(answer) and not span.strip():
                    plain.append(answer[i:end + len(opener[1])])
                    i = end + len(opener[1])
                    continue
                if end < len(answer) and span.strip() and "`" not in span and not re.search(r"(?m)^ {0,3}~{3,}", span):
                    if plain:
                        result.append(("text", "".join(plain)))
                        plain = []
                    result.append(("math", answer[start:end]))
                    i = end + len(opener[1])
                    line_start = False
                    continue
        plain.append(ch)
        line_start = False
        i += 1
    if plain:
        result.append(("text", "".join(plain)))
    return result


def _fenced_math(expression: str, number: int, reason: str = "") -> str:
    backticks = max((len(m.group()) for m in re.finditer(r"`+", expression)), default=0)
    tildes = max((len(m.group()) for m in re.finditer(r"~+", expression)), default=0)
    marker, longest = ("`", backticks) if backticks <= tildes else ("~", tildes)
    notice = f" ({reason}; TeX shown below)" if reason else ""
    if longest >= 100:
        return f"**Equation {number}**{notice}\n{expression}"
    fence = marker * max(3, longest + 1)
    return f"**Equation {number}**{notice}\n{fence}tex\n{expression}\n{fence}"


def split_text(value: str, limit: int = MAX_TEXT) -> list[str]:
    """Split on readable boundaries; carry open fenced blocks across chunks."""
    if not value:
        return []
    chunks, remaining = [], value
    marker, info = "", ""
    fence_pattern = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})([^\n]*)$")
    fences = [m for m in fence_pattern.finditer(value) if len(m.group(0)) <= 128]
    reserve = max((len(m.group(1)) + 1 for m in fences), default=0)
    while remaining:
        prefix = f"{marker}{info}\n" if marker else ""
        room = limit - len(prefix) - reserve
        cut = min(len(remaining), room)
        if cut < len(remaining):
            # Include the boundary so spaces/newlines in copyable source survive.
            for boundary in ("\n\n", "\n", " "):
                position = remaining.rfind(boundary, 0, room)
                if position >= room // 2:
                    cut = position + len(boundary)
                    break
        piece, remaining = remaining[:cut], remaining[cut:]
        for match in fence_pattern.finditer(piece):
            if len(match.group(0)) > 128:
                continue
            found, tail = match.groups()
            if not marker:
                marker, info = found, tail
            elif found[0] == marker[0] and len(found) >= len(marker) and not tail.strip():
                marker, info = "", ""
        suffix = f"\n{marker}" if marker else ""
        chunks.append(prefix + piece + suffix)
    return chunks


async def prepare_answer(answer: str, renderer: LatexRenderer | None, *, header: str = "", footer: str = "") -> list[Part]:
    if renderer is None or not renderer.settings.enabled or not renderer.settings.auto_render:
        return [Part(chunk) for value in (header + answer, footer) for chunk in split_text(value) if chunk.strip()] or [Part("No answer was generated.")]
    found = segments(answer)
    eligible = []
    for kind, value in found:
        if kind == "math" and len(value) <= renderer.settings.max_expression_chars and len(eligible) < renderer.settings.max_expressions_per_answer:
            eligible.append(value)
    results = await renderer.render_many(eligible, scale=renderer.settings.scale, theme=renderer.settings.theme) if eligible else []
    output = []
    pending = header
    rendered = 0
    count = 0
    for kind, value in found:
        if kind == "text":
            pending += value
            continue
        count += 1
        output.extend(Part(chunk) for chunk in split_text(pending) if chunk.strip())
        pending = ""
        if len(value) > renderer.settings.max_expression_chars or rendered >= len(results):
            output.extend(Part(chunk) for chunk in split_text(_fenced_math(value, count, "render limit reached")))
            continue
        result = results[rendered]
        rendered += 1
        if result.png:
            output.append(Part(f"**Equation {count}**", result.png, f"equation_{count}.png", value))
        else:
            output.extend(Part(chunk) for chunk in split_text(_fenced_math(value, count, result.error or "render failed")))
    output.extend(Part(chunk) for chunk in split_text(pending) if chunk.strip())
    output.extend(Part(chunk) for chunk in split_text(footer) if chunk.strip())
    return output or [Part("No answer was generated.")]


def _file(part: Part) -> discord.File:
    expression = part.expression or ""
    description = expression[:900] + ("… (truncated)" if len(expression) > 900 else "")
    return discord.File(BytesIO(part.png), filename=part.filename, description=description)


def _fallback(part: Part, reason: str) -> list[Part]:
    return [Part(chunk) for chunk in split_text(_fenced_math(part.expression or "", int(re.search(r"\d+", part.text).group()), reason))]


def _attach_allowed(guild, channel, byte_count: int) -> bool:
    if guild is None:
        return True
    member = guild.me
    return byte_count <= guild.filesize_limit and (member is None or channel.permissions_for(member).attach_files)


def _attachment_rejected(exc: Exception) -> bool:
    return isinstance(exc, discord.HTTPException) and (exc.status in (403, 413) or exc.code == 40005)


async def _send_parts(parts: list[Part], send, can_attach) -> bool:
    """Retry only a definitely rejected attachment; never replay delivered parts."""
    first = True
    for part in parts:
        fallback_parts = None
        if part.png and not can_attach(len(part.png)):
            fallback_parts = _fallback(part, "attachments unavailable")
        else:
            file = _file(part) if part.png else None
            try:
                await send(part.text, file, first)
                first = False
            except (discord.HTTPException, aiohttp.ClientError, OSError) as exc:
                if part.png and _attachment_rejected(exc):
                    fallback_parts = _fallback(part, "attachment rejected")
                else:
                    logger.warning("Discord answer delivery stopped (%s)", type(exc).__name__)
                    return False
            finally:
                if file:
                    file.close()
                    file.fp.close()
        for fallback in fallback_parts or []:
            try:
                await send(fallback.text, None, first)
                first = False
            except (discord.HTTPException, aiohttp.ClientError, OSError) as exc:
                logger.warning("Discord text fallback delivery stopped (%s)", type(exc).__name__)
                return False
    return True


async def send_study_parts(message: discord.Message, parts: list[Part]) -> bool:
    async def send(content, file, first):
        kwargs = {"allowed_mentions": MENTIONS}
        if file:
            kwargs["file"] = file
        await (message.reply if first else message.channel.send)(content, **kwargs)

    return await _send_parts(parts, send, lambda size: _attach_allowed(message.guild, message.channel, size))


async def send_interaction_parts(interaction: discord.Interaction, parts: list[Part]) -> bool:
    async def send(content, file, first):
        if first:
            await interaction.edit_original_response(content=content, attachments=[file] if file else [], allowed_mentions=MENTIONS)
        else:
            kwargs = {"allowed_mentions": MENTIONS}
            if file:
                kwargs["file"] = file
            await interaction.followup.send(content, **kwargs)

    def can_attach(size):
        upload_limit = getattr(interaction, "filesize_limit", None)
        if isinstance(upload_limit, int) and size > upload_limit:
            return False
        permissions = getattr(interaction, "app_permissions", None)
        if isinstance(permissions, discord.Permissions) and not permissions.attach_files:
            return False
        return _attach_allowed(interaction.guild, interaction.channel, size)

    return await _send_parts(parts, send, can_attach)
