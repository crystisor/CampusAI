import asyncio
import sys
import os
import shutil
import base64
from io import BytesIO

import pytest
from PIL import Image

from src.bot.latex_renderer import LatexRenderer, HELPER
from src.config import LatexSettings


def process_renderer(tmp_path, monkeypatch, program, **settings):
    helper = tmp_path / "helper with spaces.py"
    helper.write_text(program, encoding="utf-8")
    monkeypatch.setattr("src.bot.latex_renderer.HELPER", helper)
    renderer = LatexRenderer(LatexSettings(node_executable=sys.executable, **settings))
    renderer.available = True
    return renderer


@pytest.mark.asyncio
async def test_missing_node_falls_back():
    renderer = LatexRenderer(LatexSettings(node_executable="missing-campusai-node-executable"))
    assert not await renderer.probe()
    assert (await renderer.render_many(["x"], scale=1, theme="light"))[0].error == "unavailable"


@pytest.mark.asyncio
async def test_busy_falls_back_without_queue():
    renderer = LatexRenderer(LatexSettings())
    renderer.available = True
    await renderer._lock.acquire()
    try:
        assert (await renderer.render_many(["x"], scale=1, theme="light"))[0].error == "busy"
    finally:
        renderer._lock.release()


@pytest.mark.asyncio
async def test_timeout_reaps_child_and_releases_capacity(tmp_path, monkeypatch):
    helper = tmp_path / "slow helper.py"
    helper.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    monkeypatch.setattr("src.bot.latex_renderer.HELPER", helper)
    renderer = LatexRenderer(LatexSettings(node_executable=sys.executable, timeout_seconds=0.1))
    renderer.available = True
    result = await renderer.render_many(["x"], scale=1, theme="light")
    assert result[0].error == "timeout"
    assert not renderer._children
    assert not renderer._lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["[]", "null", "{}", "not json", '{"version":1,"results":[{"ok":false,"error":{}}]}', '{"version":1,"results":[{"ok":true,"png_base64":"bad","width":1,"height":1}]}'])
async def test_malformed_protocol_disables_renderer(tmp_path, monkeypatch, output):
    renderer = process_renderer(tmp_path, monkeypatch, f"print({output!r})")
    result = await renderer.render_many(["x"], scale=1, theme="light")
    assert result[0].error == "protocol_error"
    assert not renderer.available and not renderer._children


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_output_overflow_kills_and_reaps_child(tmp_path, monkeypatch, stream):
    renderer = process_renderer(tmp_path, monkeypatch, f"import sys, time\nsys.{stream}.write('x' * (7 * 1024 * 1024))\nsys.{stream}.flush()\ntime.sleep(5)\n")
    result = await asyncio.wait_for(renderer.render_many(["x"], scale=1, theme="light"), 3)
    assert result[0].error == "protocol_error"
    assert not renderer._children and not renderer._lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancellation_and_shutdown_reap_real_child(tmp_path, monkeypatch, shutdown):
    renderer = process_renderer(tmp_path, monkeypatch, "import time\ntime.sleep(5)\n")
    task = asyncio.create_task(renderer.render_many(["x"], scale=1, theme="light"))
    async with asyncio.timeout(3):
        while not renderer._children:
            await asyncio.sleep(0.01)
        child = next(iter(renderer._children))
        assert (await renderer.render_many(["y"], scale=1, theme="light"))[0].error == "busy"
        # Reaching here also proves that the child does not block the event loop.
        if shutdown:
            await renderer.close()
            assert (await task)[0].error == "unavailable"
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert child.returncode is not None
    assert not renderer._children and not renderer._lock.locked()


def test_configured_size_limit_is_per_expression():
    buffer = BytesIO()
    Image.new("RGB", (20, 10)).save(buffer, format="PNG")
    renderer = LatexRenderer(LatexSettings(max_width=10))
    renderer.available = True
    result = renderer._decode({"ok": True, "png_base64": base64.b64encode(buffer.getvalue()).decode(), "width": 20, "height": 10})
    assert result.error == "too_large"
    assert renderer.available


@pytest.mark.asyncio
async def test_nonzero_exit_disables_service(tmp_path, monkeypatch):
    renderer = process_renderer(tmp_path, monkeypatch, "raise SystemExit(2)")
    assert (await renderer.render_many(["x"], scale=1, theme="light"))[0].error == "unavailable"
    assert not renderer.available


@pytest.mark.parametrize("settings", [
    {"scale": float("nan")}, {"timeout_seconds": float("inf")}, {"max_expression_chars": 0},
    {"max_expressions_per_answer": 5}, {"max_png_bytes": 1048577}, {"theme": "transparent"},
])
def test_configuration_preserves_hard_limits(settings):
    with pytest.raises(ValueError):
        LatexSettings(**settings)


@pytest.mark.asyncio
@pytest.mark.latex_real
@pytest.mark.skipif(not (HELPER.parent / "node_modules").exists() or not shutil.which(os.getenv("LATEX_TEST_NODE", "node")), reason="Node and npm ci required")
async def test_real_renderer_fixtures_and_sibling_failure():
    renderer = LatexRenderer(LatexSettings(node_executable=os.getenv("LATEX_TEST_NODE", "node")))
    assert await renderer.probe()
    formulas = [
        r"\int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}",
        r"\begin{aligned}a+b &= 10 \\ 2a-b &= 5\end{aligned}",
        r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}",
        r"\frac{1}{",
    ]
    results = await renderer.render_many(formulas, scale=2, theme="dark")
    assert all(result.png and result.width > 0 and result.height > 0 for result in results[:3])
    assert results[3].error == "syntax"
    more = await renderer.render_many([
        r"\begin{cases}x^2 & x\geq0 \\ -x & x<0\end{cases}",
        r"\text{mass}\;m=2\,\mathrm{kg}",
        r"\unknowncommand{x}",
        r"\require{html}",
    ], scale=1, theme="light")
    assert more[0].png and more[1].png
    assert more[2].error == "syntax"
    assert more[3].error == "unsupported"
    repeated = await renderer.render_many([r"x+y", r"x+y"], scale=1, theme="light")
    assert repeated[0].png == repeated[1].png
    isolated = await renderer.render_many([
        r"\DeclareMathOperator{\campusfoo}{foo}\campusfoo(x)", r"\campusfoo(x)",
        r"x\label{a}", r"y\label{a}",
    ], scale=1, theme="light")
    assert isolated[0].png and isolated[1].error == "syntax"
    assert isolated[2].png and isolated[3].png
    limits = await renderer.render_many([r"\rule{100000em}{1em}", r"\def\x{\x}\x", "x" * 2001, "x"], scale=1, theme="light")
    assert [r.error for r in limits[:3]] == ["too_large", "unsupported", "invalid_input"]
    assert limits[3].png and renderer.available
    recovery = await renderer.render_many([r"\frac{1}{", "x", r"\begin{missing}x\end{missing}", r"\mathbb{R}\to\mathbb{C}"], scale=1, theme="light")
    assert recovery[0].error == "syntax" and recovery[1].png
    assert recovery[2].error == "syntax" and recovery[3].png
    await renderer.close()
