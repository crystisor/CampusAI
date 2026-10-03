# Implementation Plan: LaTeX Output for CampusAI in Discord

Status: implemented locally, with automated verification; live Discord acceptance remains pending.
Reviewed against the local repository and upstream source on 2026-10-02.

Implementation verified on 2026-10-03 with Node 24.21.0, MathJax 4.1.3,
Sharp 0.35.5, and Python 3.14. The checklist below retains unchecked live-server
items; local rendering and mocked Discord tests do not establish live acceptance.
The complete Python suite passed (91 tests), including the `latex_real` fixtures;
Python/JavaScript syntax checks and `git diff --check` also passed.

## 1. Goal and scope

The primary goal is for CampusAI to answer ordinary user questions and automatically use LaTeX for mathematical expressions when it improves the explanation. Users should not need to request LaTeX, supply TeX source, or run a rendering command. The answer-generating model chooses suitable mathematical notation; the bot detects the marked expressions and renders them before sending the answer.

Support this behavior in:

1. Answers to messages in subject-bound study channels.
2. Answers produced by the existing `/ask` command.
3. As a secondary convenience, a new `/latex code:<expression> [scale] [theme]` command that renders supplied TeX without invoking the LLM. This command is not a prerequisite for automatic math in answers.

Keep explanations, citations, and variable definitions as Discord text. Discord cannot place an attachment inside a sentence; inline expressions become labeled image blocks next to the surrounding explanation.

For example, a student asking "How do I solve a quadratic equation?" receives an explanation with a typeset quadratic formula and relevant steps. A question such as "What is the purpose of this course?" receives a normal text answer without unnecessary formulas or renderer calls. Do not require formula-related keywords to enable generated-equation rendering; the existing keyword heuristic applies only to selecting source PDF pages.

Use the existing Python/discord.py bot with a small local Node.js helper for MathJax and Sharp. Rendering must work offline after dependency installation. Support mathematical TeX, including fractions, integrals, matrices, cases, and aligned equations. Full LaTeX documents, arbitrary packages, TikZ, and shell-based TeX compilation are outside this feature.

## 2. Analysis of the reference implementation

Reference: [alvesvaren/latex-bot-ts](https://github.com/alvesvaren/latex-bot-ts), inspected on `main` with Git tree SHA `18cd37ba9abbc52fe174a419585105765adb2594`. Source links below follow `main`; the tree SHA records the inspected snapshot.

- [commands/latex.ts](https://github.com/alvesvaren/latex-bot-ts/blob/main/commands/latex.ts) initializes MathJax at module load, wraps input in white-colored `aligned` math, converts TeX to SVG, rasterizes with Sharp, and returns an embedded PNG. It exposes brightness and scale.
- [index.ts](https://github.com/alvesvaren/latex-bot-ts/blob/main/index.ts) defers interactions in the shared dispatcher before executing the command. It registers all loaded commands and catches command errors.
- [package.json](https://github.com/alvesvaren/latex-bot-ts/blob/main/package.json) uses `mathjax ^3.2.2`, `sharp ^0.32.1`, and `discord.js ^14.11.0`. It declares MIT; the inspected tree has no standalone LICENSE file. This plan adapts the architecture without copying its implementation. Preserve attribution and establish applicable license text if reusing code.
- [README](https://github.com/alvesvaren/latex-bot-ts/blob/main/README.md) describes this as an older public implementation; some live-bot features are absent.

Retain its TeX -> SVG -> PNG approach and prompt interaction acknowledgment. Adapt the integration, output handling, dependency versions, and resource limits to CampusAI. The upstream command does not implement automatic LLM-answer rendering, bounded subprocess execution, or per-expression failure recovery.

### Problems in the original local plan

| Original assumption | Repository-specific correction |
| --- | --- |
| Build TypeScript commands and `src/index.ts` | Add Python cogs to the existing `StudyBot` |
| Install another Discord client library | Keep `discord.py`; Node only renders images |
| Slash command alone enables math answers | Integrate study replies and `/ask`, and update the prompt that currently forbids LaTeX |
| Unpinned `mathjax` installation with v3 code | Use one consistent MathJax major/API and lock tested dependencies |
| Transparent white text and brightness | Use controlled foreground/background themes readable in either Discord theme |
| Always wrap input in `aligned` | Accept complete supported environments without an extra wrapper |
| Presence of SVG means success | Capture parser failures and validate PNG output |
| Register only the new command with REST | Load the cog before the existing complete `tree.sync()` |
| Resize without resource limits | Bound input, process lifetime, SVG/raster dimensions, and output bytes |

## 3. Current code and integration points

| File / symbol | Current behavior | Planned change |
| --- | --- | --- |
| `src/bot/bot.py`: `StudyBot.setup_hook` | Loads study/admin cogs and syncs the command tree | Own one shared renderer; probe it and load the new cog before sync |
| `src/bot/cogs/study_chat.py`: `StudyChatCog.on_message` | Calls RAG in bound guild channels, cleans output, sends chunked text | Render marked math before text chunking |
| `src/bot/cogs/admin.py`: `AdminCog.ask_cmd` | Defers, calls RAG, sends text | Use the same math preparation and delivery path |
| `src/rag/pipeline.py`: `SYSTEM_PROMPT_TEMPLATE` | Explicitly prohibits LaTeX commands and delimiters | Select math instructions according to rendering capability |
| `src/bot/cogs/study_chat.py`: `source_formula_page`, `render_formula_page` | Selects and renders an original PDF page using Poppler | Preserve this evidence path alongside generated equations |
| `src/bot/cogs/study_chat.py`: `format_references` | Lists all supplied sources, groups PDF pages, and includes course numbers when present | Preserve its output as a separate text part outside math detection |
| `src/config.py`: `AppConfig`, `load_config` | Pydantic settings from YAML | Add validated `LatexSettings` |
| `src/web/static/js/dashboard.js` and web template | Browser-only KaTeX preview | No change required; this does not render Discord attachments |
| `requirements.txt` | Python runtime, Pillow, pytest | No additional Python rendering dependency expected |
| `docker-compose.yml` | Qdrant only | Install the renderer on the host running the Python bot |

The existing graphify graph was used for orientation, then verified against source files because some graph locations predate current code.

## 4. Proposed architecture

~~~text
StudyChatCog.on_message                 AdminCog.ask_cmd
           |                                  |
           +---- RAGPipeline.process_query ---+
                              |
                    clean_llm_response
                              |
                parse text / math segments
                              |
                  shared LatexRenderer <---- LatexCog /latex
                              |
                 bounded local Node process
                              |
            MathJax TeX -> standalone SVG -> Sharp PNG
                              |
            ordered Discord text / image parts
                              |
             citations and selected source PDF page
~~~

### New modules

- `src/bot/latex_renderer.py`: availability probe, admission control, subprocess lifecycle, typed failures, validated PNG bytes. No Discord or LLM calls.
- `src/bot/math_messages.py`: pure segmentation and output preparation, with small delivery helpers for replies and deferred interactions.
- `src/bot/cogs/latex.py`: command declaration, validation, immediate defer, rendering, and response.
- `tools/latex-renderer/render.mjs`: local JSON-in/JSON-out executable. No HTTP service or Discord credentials.

Suggested renderer interface:

~~~python
async def probe(self) -> bool: ...
async def render_many(self, expressions: list[str], *, scale: float, theme: str) -> list[RenderResult]: ...
async def close(self) -> None: ...
~~~

Each `RenderResult` contains PNG bytes and dimensions, or a controlled category: `invalid_input`, `syntax`, `unsupported`, `too_large`, `busy`, `timeout`, `unavailable`, or `protocol_error`. The message layer retains original TeX for fallback.

### Process and protocol

Use one short-lived Node process per batch of up to four expressions. Initialize MathJax once per batch, then render sequentially. This incurs startup cost per answer but gives Python a process it can terminate on timeout. Measure latency before introducing a persistent worker or cache.

Launch via `asyncio.create_subprocess_exec`, passing an argument array and an absolute helper path resolved from `BASE_DIR`. Send UTF-8 JSON on stdin. Never interpolate TeX into shell commands, filenames, executable arguments, or JavaScript source. Test paths containing spaces.

Example input:

~~~json
{"version":1,"expressions":["E=mc^2","\\frac{a}{b}"],"scale":2,"theme":"light"}
~~~

Return one JSON document with `version` and an ordered `results` array. Each entry contains either `ok: true`, `png_base64`, `width`, and `height`, or `ok: false` and a controlled error category. A syntax failure must not discard successful sibling expressions; a batch crash or deadline falls back to source text for the whole batch.

Reserve stdout for protocol data and stderr for bounded diagnostics. Validate schemas, result counts, base64, PNG decoding, byte counts, and dimensions on the Python side. Initial fixed protocol caps: 64 KiB input, 6 MiB stdout, and 16 KiB stderr. Enforce total stream caps while reading, not just after buffering.

Start with one active batch globally and immediate `busy` fallback instead of an unbounded waiting queue. Put a deadline around the complete child lifetime. On timeout, cancellation, output overflow, or shutdown, kill and await the child, close pipes, and release capacity. Track active children in the shared renderer; clean up from `StudyBot.close()` before completing the parent close.

An asyncio timeout alone does not terminate a subprocess. Windows needs a subprocess-capable event loop. These behaviors require real process tests. [Python asyncio subprocess documentation](https://docs.python.org/3/library/asyncio-subprocess.html)

## 5. MathJax and Sharp implementation

### Dependencies

Add an isolated `tools/latex-renderer/package.json` and committed `package-lock.json`. Use JavaScript ESM, `@mathjax/src` major 4, and `sharp`. Choose exact compatible releases during implementation, save exact versions, and document the tested supported Node LTS version in `engines` and the README. Install and lock any font package required by that MathJax configuration.

Deploy the committed set with:

~~~powershell
npm ci --prefix tools/latex-renderer
~~~

Install on each deployment platform; do not copy Windows `node_modules` onto Linux. Sharp selects platform-specific native binaries and needs optional dependencies installed. [Sharp installation](https://sharp.pixelplumbing.com/install/)

Do not add TypeScript, `@types/sharp`, a browser runtime, or another Discord SDK. The Python bridge uses the standard library and existing Pillow. The existing `.gitignore` already excludes `node_modules/`.

### MathJax setup and input policy

1. Configure a local loader, liteDOM, TeX input, SVG output, and a locally installed font before startup. Await startup and use asynchronous conversion for local font loading. Call `MathJax.done()` at process completion where applicable. Do not combine v3 `init()` examples with v4 package paths. [MathJax Node setup](https://docs.mathjax.org/en/latest/server/components.html)
2. Start from `input/tex-base`; explicitly load AMS and enable only `base` and `ams`. Exclude `require`, `autoload`, HTML/link/image extensions, and user-selected loader paths. Unsupported commands fail without fetching packages. [MathJax TeX extensions](https://docs.mathjax.org/en/latest/input/tex/extensions.html)
3. Set finite buffer and expansion limits supported by the pinned version. Configure `formatError` to report/throw parse failures instead of accepting an error image. Include unknown commands and malformed environments in tests. [TeX input options](https://docs.mathjax.org/en/latest/options/input/tex.html)
4. Isolate TeX state between expressions: macro definitions, labels, and numbering must not leak. Reuse loaded modules/font data, but reset or recreate per-expression input/document state and test the behavior.
5. Use `svg.fontCache: 'none'` to make each SVG independent. Serialize the SVG element through the loaded adaptor, with valid namespace, finite dimensions, and viewBox. [MathJax SVG options](https://docs.mathjax.org/en/latest/options/output/svg.html)

The answer parser removes delimiters and passes bare TeX. For `/latex`, accept bare TeX or strip one matching outer `$$...$$`, `\[...\]`, or `\(...\)` pair. Preserve backslashes and interior whitespace. Require explicit `aligned` for multiple alignment rows; accept complete `pmatrix`, `cases`, and other supported environments directly.

### Appearance and resource checks

- Default `light`: dark text on an opaque near-white background. Optional `dark`: near-white text on an opaque dark background. Add padding. Replace brightness with this theme choice.
- Apply controlled foreground color to generated SVG. Theme is an enum, not arbitrary CSS or TeX.
- Convert MathJax relative dimensions to pixels using the selected font metrics; preserve viewBox/aspect ratio. Missing or invalid dimensions are errors, not guessed defaults.
- Reject SVG above 1 MiB and out-of-bounds target dimensions before rasterization.
- Rasterize at the requested resolution; account for scale before creating pixels, rather than enlarging a low-resolution PNG.
- Use Sharp `limitInputPixels` as an additional guard; verify final width, height, pixel count, and PNG size. Sharp `density` controls SVG rasterization resolution. [Sharp input options](https://sharp.pixelplumbing.com/api-constructor/)
- Return images in memory. No persistent cache or shared temporary output files are needed.

The child process provides termination isolation, not an OS security sandbox. Never send user or model TeX to `latex`, `pdflatex`, or a shell. Do not log full formulas, answers, or the child environment.

## 6. Configuration and lifecycle

Add `LatexSettings` and `AppConfig.latex` with `Field(default_factory=LatexSettings)`. Add matching YAML defaults:

~~~yaml
latex:
  enabled: true
  auto_render: true
  node_executable: "node"
  theme: "light"
  scale: 2.0
  timeout_seconds: 10.0
  max_expression_chars: 2000
  max_expressions_per_answer: 4
  max_width: 2048
  max_height: 1024
  max_pixels: 2097152
  max_png_bytes: 1048576
~~~

Validate finite positive limits, theme membership, and scale in `[0.5, 3.0]`. Preserve hard ceilings on expression count, dimensions, and bytes so configuration cannot exceed protocol bounds. These are initial product limits to validate against the fixtures.

Create the shared renderer in `StudyBot.setup_hook` and probe it by rendering a small known formula under the same deadline. Missing Node, dependencies, fonts, or Sharp marks it unavailable and produces an actionable startup log. Study/admin cogs and command sync must still load. Do not install dependencies at startup.

Automatic rendering capability is `enabled AND auto_render AND available`; `/latex` needs `enabled AND available`. A process-level failure should mark availability false until a successful bounded probe or restart. A syntax, input, size, or busy failure must not disable the service globally.

## 7. Automatic math in study replies and /ask

### Prompt changes

Extend `RAGPipeline.process_query` with keyword-only `render_math: bool = False`. Both Discord answer paths pass the renderer's automatic capability; existing callers retain plain-text behavior by default. This flag permits mathematical rendering; it does not require every answer to contain an equation. Apply the presentation instructions consistently to DIRECT, RAG, WEB, and HYBRID answers without introducing another routing decision or a separate LLM call.

Select only the math-presentation instruction block:

- Disabled/unavailable: retain current plain-text/Unicode math instructions.
- Enabled: answer the user's question first and choose LaTeX when equations, derivations, fractions, matrices, or other mathematical notation make the answer clearer. No explicit request for LaTeX is needed. Use plain text when mathematical typesetting adds no value. Put expressions intended for rendering inside `$$...$$`, keep simple variables in prose, and use explicit `aligned` for aligned rows. Keep explanations, conditions, units, and citations outside equations. Do not put intended rendered math inside code fences or generate a complete LaTeX document.

Preserve evidence, OCR uncertainty, citation, and private-reasoning rules, including the current `course_number` citation metadata. Rendering changes appearance, not correctness. Do not invoke the LLM again merely because an image failed.

The template currently calls `.format(subject_name=...)`. Insert math instructions as a substitution value or otherwise protect literal TeX braces from format-field interpretation.

### Segmentation contract

Implement a small stateful scanner in `math_messages.py`:

- Recognize `$$...$$`, `\[...\]`, and `\(...\)` outside code spans and fenced blocks, honoring escaping and delimiter precedence.
- Leave single `$` untouched to avoid interpreting prices as math; the prompt must not request single-dollar delimiters.
- Preserve unmatched delimiters, empty spans, and unsupported forms as literal text. An unmatched opener must not swallow the remaining answer.
- Preserve inline code and backtick/tilde fences, including fenced LaTeX examples.
- Parse only the cleaned answer, before Discord chunking. Exclude the generated subject header and references footer from math detection.
- Assign equation numbers in occurrence order. Attempt up to four eligible expressions; keep oversized and excess expressions as copyable text.

For example, `The energy is \(E=mc^2\). Here m is mass.` becomes text referring to Equation 1, the labeled equation image, and the remaining explanation in order.

### Message preparation and delivery

Build ordered text/image parts. Give each PNG a name such as `equation_1.png`, a matching caption, and a bounded attachment description containing source TeX. Mark truncation in descriptions explicitly.

Use `discord.File(BytesIO(png), filename=..., description=...)`. A bare attachment is sufficient. If using an embed, its `attachment://` name must match and Embed Links permission must be handled. Allocate a new buffer/file for each send attempt and close it afterward.

Split text at paragraph/newline/word boundaries into at most 1,950 characters including inserted labels/fence markers, with a hard split for long tokens. Close/reopen code fences when necessary. Never split math source before parsing. Preserve the full output of the existing `format_references(top_contexts)` in a separate text part when necessary, including grouped pages, course numbers, and web sources. Pass this footer into the shared preparation helper from the cog so the helper does not import its caller.

- Study replies: first part replies to the triggering message; subsequent parts go to the same channel in order.
- `/ask`: retain immediate defer, edit the original response with the first part, then use interaction followups.
- Send one generated equation per image part. Check bytes against both configured limits and the destination's applicable upload limit; do not assume a universal Discord upload cap.
- Use `discord.AllowedMentions.none()` for generated and fallback content.

On a render failure, replace that equation with safely fenced, copyable TeX and a brief notice. Handle backticks inside the source when selecting fences. Other successful parts remain available.

If Attach Files is missing, use text fallback. On a definite attachment rejection, retry only that unsent part as text. Do not resend already delivered parts or blindly retry an ambiguous network failure. If text delivery is also forbidden, log and stop.

### Existing PDF formula behavior

Preserve `source_formula_page`, its path containment checks, and `render_formula_page`. Send the selected page independently, labeled **Source PDF page** with available document/page metadata. Check its size separately and retain citations if its upload fails.

The selector currently uses formula-related query text and the first eligible retrieved page; it does not prove that a generated formula is an exact transcription. Never label generated MathJax output as original PDF evidence. Keep source-page selection in study replies; extending it to `/ask` is not required for this feature.

## 8. New /latex command

Create a `LatexCog` using `discord.app_commands.command` and the current cog `setup` convention:

- `code`: required string, constrained by the input limit and revalidated after normalization.
- `scale`: optional number in `[0.5, 3.0]`, default from configuration.
- `theme`: optional `light`/`dark` choice, default from configuration.

Make the initial command guild-only, available without a subject binding. It must not invoke RAG, routing, web search, or Ollama.

Immediately call `await interaction.response.defer(thinking=True)` before rendering. Discord requires initial acknowledgment within three seconds. [Discord interaction lifecycle](https://docs.discord.com/developers/interactions/receiving-and-responding)

Complete the response with `interaction.edit_original_response(attachments=[discord.File(...)], ...)`. Editing uses `attachments`, not `file`. Report controlled errors using that deferred response, without attempting a second initial acknowledgment. [discord.py interaction API](https://discordpy.readthedocs.io/en/stable/interactions/api.html)

Load `src.bot.cogs.latex` before the existing `tree.sync()`. Verify `/ask`, `/bind`, `/unbind`, `/status`, and `/subjects` remain registered alongside `/latex`. No second client or extra message listener is needed.

## 9. Implementation order and file checklist

1. Build the isolated Node helper, lock dependencies, and verify real mathematical fixtures offline.
2. Add validated settings, the Python bridge, bounded cleanup, and the availability probe.
3. Implement the scanner, ordered message preparation, chunking, and fallback.
4. Add bot ownership/shutdown and the new cog through existing command sync.
5. Add capability-aware prompt instructions and integrate both answer paths.
6. Verify source-page coexistence, update documentation, and complete acceptance checks.

| File | Planned work |
| --- | --- |
| `tools/latex-renderer/package.json`, `package-lock.json`, `render.mjs` | New helper and reproducible dependencies |
| `src/bot/latex_renderer.py` | Shared async renderer service |
| `src/bot/math_messages.py` | Parsing, prepared parts, delivery helpers |
| `src/bot/cogs/latex.py` | Explicit rendering command |
| `src/bot/bot.py` | Startup/probe, shared ownership, cleanup, cog loading |
| `src/bot/cogs/study_chat.py` | Automatic rendering and preserved source-page path |
| `src/bot/cogs/admin.py` | `/ask` uses shared math output |
| `src/rag/pipeline.py` | Capability parameter and math instruction selection |
| `src/config.py`, `config.yaml` | Settings and defaults |
| `tests/test_latex_renderer.py`, `tests/test_math_messages.py`, `tests/test_latex_commands.py` | New focused tests |
| `tests/test_rag_pipeline.py`, `tests/test_formula_page.py` | Prompt and evidence regressions |
| `README.md` | Installation, syntax, limits, availability, troubleshooting |

No ingestion, Qdrant, dashboard, or data migration is required.

## 10. Verification and acceptance

### Real rendering fixtures

| TeX input | Expected |
| --- | --- |
| `E = mc^2` | Sharp padded PNG in both themes |
| `\int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}` | Full integral/fraction/superscripts without clipping |
| `\begin{aligned}a+b &= 10 \\ 2a-b &= 5\end{aligned}` | Two aligned rows |
| `\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}` | Matrix entries and parentheses |
| `\begin{cases}x^2 & x\geq0 \\ -x & x<0\end{cases}` | Cases and inequalities |
| `\text{mass}\;m=2\,\mathrm{kg}` | Text, spacing, units |
| `\frac{1}{` or unknown command | Controlled failure, not an error image reported as success |
| `\require{html}` or resource commands | Rejected without TeX-initiated network/filesystem loading |
| Recursive definitions, huge dimensions, oversized input | Bounded failure and responsive bot |

### Automated coverage

- Scanner: mixed prose/math, multiple equations, escaping, prices, malformed/empty delimiters, code spans, backtick/tilde fences, and expression limits.
- Delivery: all parts fit message limits; equation order and references survive chunking; plain answers spawn no renderer; disabled mode preserves text; missing permissions and upload rejections fall back correctly.
- Bridge: busy admission, missing executable, failed probe, nonzero exit, malformed protocol/PNG, output overflow, timeout, cancellation, and shutdown. Prove children are reaped and capacity released.
- Real renderer: decode PNGs and check dimensions; preserve a valid sibling after syntax failure; prove state isolation; verify a heartbeat coroutine runs during child execution.
- Cogs with mocked Discord: defer before rendering, correct attachment arguments, controlled errors, no LLM calls for `/latex`, and both automatic output paths.
- Prompt: enabled/disabled selection across all four routing modes, permission to use math without requiring it, literal TeX braces, and existing evidence/citation/reasoning-filter regressions.
- PDF path: retain selection, containment, and complete-reference-formatting tests; verify source-page and generated-equation attachments can coexist.

Unit tests must not require Discord, Ollama, Qdrant, or Node. Mark real-renderer tests separately: they may skip in a Python-only developer environment, but must run with the locked dependencies before shipping.

Run focused tests during implementation, then the existing suite:

~~~powershell
python -m pytest tests/test_latex_renderer.py tests/test_math_messages.py tests/test_latex_commands.py tests/test_rag_pipeline.py tests/test_formula_page.py
python -m pytest tests/
~~~

### Live acceptance checklist

- [ ] `/latex` renders fixtures and handles malformed TeX.
- [ ] Bound-channel answers and `/ask` both produce equations with readable explanations.
- [ ] An ordinary question such as "How do I solve a quadratic equation?" receives an explanation and rendered math without mentioning LaTeX or using `/latex`.
- [ ] Questions that do not benefit from mathematical notation receive normal text answers and start no rendering process.
- [ ] Long answers preserve order, citations, and selected PDF pages.
- [ ] Desktop/mobile and light/dark Discord views show legible, unclipped images.
- [ ] Missing dependencies or Attach Files permission still allow text answers.
- [ ] Concurrent requests and a hung renderer leave the bot responsive without orphaned children.
- [ ] Offline rendering works after installation; all existing commands remain registered.
- [ ] The suite passes and real rendering is verified on the deployment host.

Completion means all three output paths work and failures preserve useful answers. Persistent workers, caching, broader package support, and document rendering are follow-up work only if measured requirements justify them.
