from unittest.mock import AsyncMock, Mock

import pytest
import discord

from src.bot.latex_renderer import RenderResult
from src.bot.math_messages import Part, prepare_answer, segments, split_text, send_study_parts, send_interaction_parts
from src.config import LatexSettings


COURSE_FORMULA = r"QLTY = 100 \times \frac{\sum_{i=1}^{#TUS} QLTY_i}{#TUS}"


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.asyncio
async def test_course_formula_renders_with_literal_counts(wrapped):
    renderer = Mock(settings=LatexSettings(), render_many=AsyncMock(return_value=[RenderResult(png=b"png")]))
    equation = "$$" + COURSE_FORMULA + "$$" if wrapped else COURSE_FORMULA
    parts = await prepare_answer("Formula:\n" + equation + "\nExplanation.", renderer)
    assert [part.text for part in parts] == ["Formula:\n", "**Equation 1**", "\nExplanation."]
    assert parts[1].png and parts[1].expression == COURSE_FORMULA
    renderer.render_many.assert_awaited_once_with(
        [COURSE_FORMULA.replace("#", r"\#")], scale=2.0, theme="light")


@pytest.mark.parametrize("answer", [
    "```tex\n" + COURSE_FORMULA + "\n```",
    "~~~\n" + COURSE_FORMULA + "\n~~~",
    "`" + COURSE_FORMULA + "`",
    "Explain " + COURSE_FORMULA,
    "Price = $5 and " + COURSE_FORMULA,
    r"x = \frac{1}{2} $$unfinished",
    "Ordinary prose.",
])
def test_bare_equation_detection_preserves_non_math(answer):
    assert segments(answer) == [("text", answer)]


@pytest.mark.asyncio
async def test_already_escaped_count_is_not_double_escaped():
    renderer = Mock(settings=LatexSettings(), render_many=AsyncMock(return_value=[RenderResult(png=b"png")]))
    formula = COURSE_FORMULA.replace("#", r"\#")
    await prepare_answer("$$" + formula + "$$", renderer)
    assert renderer.render_many.call_args.args == ([formula],)


@pytest.mark.latex_real
@pytest.mark.asyncio
async def test_course_formula_produces_real_png():
    from src.bot.latex_renderer import LatexRenderer

    renderer = LatexRenderer(LatexSettings())
    try:
        if not await renderer.probe():
            pytest.skip("Local Node renderer dependencies unavailable")
        parts = await prepare_answer(COURSE_FORMULA, renderer)
        assert len(parts) == 1
        assert parts[0].png.startswith(b"\x89PNG\r\n\x1a\n")
        assert parts[0].expression == COURSE_FORMULA
    finally:
        await renderer.close()


def test_scanner_preserves_code_prices_and_malformed_math():
    answer = "Price $5, `$$code$$`, and\n~~~tex\n$$sample$$\n~~~\nReal \\(x+1\\). Broken $$x"
    assert segments(answer) == [
        ("text", "Price $5, `$$code$$`, and\n~~~tex\n$$sample$$\n~~~\nReal "),
        ("math", "x+1"),
        ("text", ". Broken $$x"),
    ]


@pytest.mark.parametrize("answer,expressions", [
    (r"Price $4.50 and \$5. $$x$$ and \[y\] and \(z\)", ["x", "y", "z"]),
    (r"Escaped \\(x\\) and \$$literal\$$", []),
    ("$$$$ $$ $$ then $$x$$", ["x"]),
    ("```tex\n$$example$$\n```\n$$real$$", ["real"]),
    ("`` ` $$example$$ `` and $$real$$", ["real"]),
    (r"Unmatched \[ prose and \(valid\)", ["valid"]),
    ("```\n$$unclosed fence$$", []),
])
def test_scanner_contract(answer, expressions):
    assert [value for kind, value in segments(answer) if kind == "math"] == expressions


@pytest.mark.asyncio
async def test_answer_orders_equations_and_preserves_footer():
    renderer = AsyncMock()
    renderer.auto_available = True
    renderer.settings = LatexSettings()
    renderer.render_many.return_value = [RenderResult(png=b"png", width=20, height=20)]
    parts = await prepare_answer("First $$x=1$$ then text.", renderer, header="Header\n", footer="\nSources: page 3")
    assert [part.text for part in parts] == ["Header\nFirst ", "**Equation 1**", " then text.", "\nSources: page 3"]
    renderer.render_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_plain_answer_never_calls_renderer():
    renderer = AsyncMock()
    renderer.auto_available = True
    renderer.settings = LatexSettings()
    parts = await prepare_answer("Ordinary answer.", renderer)
    assert [p.text for p in parts] == ["Ordinary answer."]
    renderer.render_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_long_failed_expression_stays_within_discord_limit():
    renderer = AsyncMock()
    renderer.auto_available = True
    renderer.settings = LatexSettings()
    renderer.render_many.return_value = [RenderResult(error="syntax")]
    parts = await prepare_answer("$$" + "x" * 2000 + "$$", renderer)
    assert all(len(part.text) <= 1950 for part in parts)
    source_lines = [line for part in parts for line in part.text.splitlines() if line and set(line) == {"x"}]
    assert sum(map(len, source_lines)) == 2000


def test_long_text_parts_fit_limit():
    assert all(len(chunk) <= 1950 for chunk in split_text("word " * 1000))


@pytest.mark.asyncio
async def test_missing_upload_capacity_sends_copyable_tex():
    guild = Mock(filesize_limit=1, me=None)
    channel = Mock(send=AsyncMock())
    message = Mock(guild=guild, channel=channel, reply=AsyncMock())
    await send_study_parts(message, [Part("**Equation 1**", b"png", "equation_1.png", r"x^2")])
    assert "x^2" in message.reply.call_args.args[0]
    assert "file" not in message.reply.call_args.kwargs


@pytest.mark.asyncio
async def test_deferred_interaction_uses_original_then_followup():
    interaction = Mock(guild=None, edit_original_response=AsyncMock(), followup=Mock(send=AsyncMock()))
    await send_interaction_parts(interaction, [Part("First"), Part("Second")])
    assert interaction.edit_original_response.call_args.kwargs["content"] == "First"
    assert interaction.followup.send.call_args.args == ("Second",)
    assert "file" not in interaction.followup.send.call_args.kwargs


@pytest.mark.asyncio
async def test_rejected_attachment_retries_only_unsent_equation():
    error = discord.HTTPException(Mock(status=413, reason="too large"), {"code": 40005, "message": "too large"})
    interaction = Mock(guild=None, edit_original_response=AsyncMock(), followup=Mock(send=AsyncMock(side_effect=[error, None, None])))
    assert await send_interaction_parts(interaction, [Part("Explanation"), Part("**Equation 1**", b"png", "equation_1.png", "x"), Part("Sources")])
    interaction.edit_original_response.assert_awaited_once()
    calls = interaction.followup.send.call_args_list
    assert len(calls) == 3 and "file" in calls[0].kwargs and "file" not in calls[1].kwargs
    assert "TeX shown below" in calls[1].args[0] and calls[2].args[0] == "Sources"


@pytest.mark.asyncio
async def test_ambiguous_failure_does_not_retry_or_overwrite_delivered_text():
    interaction = Mock(guild=None, edit_original_response=AsyncMock(), followup=Mock(send=AsyncMock(side_effect=OSError("lost connection"))))
    assert not await send_interaction_parts(interaction, [Part("Explanation"), Part("**Equation 1**", b"png", "equation_1.png", "x"), Part("Sources")])
    interaction.edit_original_response.assert_awaited_once()
    interaction.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_mode_and_expression_limits():
    renderer = Mock(settings=LatexSettings(enabled=False), render_many=AsyncMock())
    parts = await prepare_answer("Keep $$x$$ literal.", renderer)
    assert parts[0].text == "Keep $$x$$ literal."
    renderer.render_many.assert_not_awaited()
    renderer.settings = LatexSettings(max_expression_chars=3, max_expressions_per_answer=1)
    renderer.render_many.return_value = [RenderResult(png=b"png")]
    parts = await prepare_answer("$$long$$ $$x$$ $$y$$", renderer)
    assert [p.filename for p in parts if p.png] == ["equation_2.png"]
    renderer.render_many.assert_awaited_once_with(["x"], scale=2.0, theme="light")
    assert "long" in parts[0].text and "y" in parts[-1].text
