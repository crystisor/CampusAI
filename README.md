# 🎓 CampusAI — Discord Bot Powered by a Local AI Stack

A Discord-based academic assistant that answers course-related questions using a **fully local AI stack**.

CampusAI combines **LLM inference, RAG, document processing, web search, reranking, and intent routing** to provide context-aware answers directly inside Discord.

The system is designed around privacy, local inference, and efficient retrieval — allowing academic course material to remain on the local machine.

---

## ✨ Features

* 🤖 **Discord AI Assistant** — Ask questions directly from Discord.
* 🧠 **Local LLM inference** — Powered by `gemma4_e2b_q8:latest`.
* 📚 **RAG-based course knowledge** — Course material is indexed in Qdrant.
* 🔎 **Semantic search** — Uses `bge-m3` embeddings.
* 🎯 **Reranking** — Uses `bge-reranker-v2-m3` to select the most relevant context.
* 🌐 **Web search** — Supports questions requiring information outside the indexed course material.
* 🧭 **Intent routing** — `Arch-Router` determines how each query should be handled.
* 👁️ **Visual document processing** — GLM-OCR transcribes scanned text; Gemma 4 interprets charts, diagrams and images on PDF pages.
* 📄 **PDF ingestion** — Upload and process course material through the web dashboard.
* 🖥️ **Management dashboard** — Monitor subjects, documents, hardware, VRAM, and vector database status.
* 🔒 **Local-first architecture** — AI inference and vector storage can run entirely on your own hardware.

---

# 🏗️ Architecture

### Temporary channel memory

Use `/start` to enable shared conversation memory in a guild text channel. The session survives bot restarts and stores accepted questions and answers in `storage/sessions.sqlite3`. Everyone in the channel shares that context. `/start` does not reset an active session. Use `/end` to delete the session and its stored transcript; messages already posted in Discord remain visible. Without an active session, ordinary questions and `/ask` keep their stateless behavior. `/ask` subject overrides are retained with each turn.

Conversation memory is currently supplied as stored conversation history; older session context can grow over time. Summaries are approximate recall, and exact recall of older details is not guaranteed.

CampusAI uses a modular architecture consisting of:

### Clickable course sources in Discord

Course-based answers include a **Sources** list with the document name and a
separate clickable link for each retrieved slide/page. For example,
`Course: Paper_Unit_2.3_Extreme_Programming — Slides/pages 16, 14` has individual
links on `16` and `14`.

The bot posts an image of each exact original PDF page in the same channel.
Clicking a page number jumps to that source message, where the image can be
opened at full size. Links remain usable while the source messages exist and
the student has channel access; no publicly hosted dashboard is required.
Page numbers use the PDF's one-based page order. Repeated references to the
same page share one preview per answer, including answers that combine course
material with web results.

The bot needs **Attach Files**, and the original PDF must remain in the subject's
`raw` storage directory. Missing or unrenderable pages, oversized attachments,
and failed uploads are labeled `preview unavailable`; the citation stays visible.
The sources list reflects the passages supplied to answer generation.

### Backend

* **Python**
* **FastAPI**
* **Uvicorn**

### Frontend

* HTML
* CSS
* JavaScript

### Vector Database

* **Qdrant** — stores embeddings and enables semantic retrieval for RAG.

### AI Model Stack

| Component           | Model                | Purpose                               |
| ------------------- | -------------------- | ------------------------------------- |
| **LLM**             | `gemma4_e2b_q8:latest` | Answer generation                     |
| **Visual Model**    | `gemma4_e2b_q8:latest` | Chart, diagram and image interpretation |
| **OCR**             | `glm-ocr-eot`         | Scanned text and formula transcription |
| **Embedding Model** | `bge-m3`             | Document and query embeddings         |
| **Reranker**        | `bge-reranker-v2-m3` | Context relevance ranking             |
| **Router**          | `Arch-Router`        | Query intent classification           |

---

# 🔀 Query Routing

Every question is first processed by **Arch-Router**, which determines the appropriate retrieval strategy.

```text
                         Discord
                    #calculus-1
                         │
                         ▼
                  ┌──────────────┐
                  │  Arch-Router │
                  │    Intent    │
                  │ Classification│
                  └───────┬──────┘
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
       DIRECT            RAG              WEB
          │               │                │
          │               ▼                ▼
          │          ┌──────────┐     Web Search
          │          │  Qdrant  │          │
          │          │  + BGE   │          │
          │          └────┬─────┘          │
          │               │                │
          │               └───────┬────────┘
          │                       ▼
          │              BGE Reranker v2
          │                       │
          └───────────────────────┤
                                  ▼
                         gemma4_e2b_q8:latest
                                  │
                                  ▼
                           Discord Reply
```

### Routing Modes

| Mode       | Description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| **DIRECT** | Sends the question directly to the local LLM.                                |
| **RAG**    | Retrieves relevant course material from Qdrant before generating the answer. |
| **WEB**    | Uses web search to retrieve external information.                            |
| **HYBRID** | Combines course material from Qdrant with external web information.          |

For RAG and web-based queries, retrieved documents are passed through `bge-reranker-v2-m3` before being sent to the LLM.

---

# 🖥️ Web Management Dashboard

CampusAI includes a minimalistic web dashboard for managing the knowledge base and monitoring the local AI infrastructure.

The dashboard provides:

### 📚 Subject Directory

Manage registered academic subjects and their associated Discord channels.

### 📄 PDF Upload & Processing

Upload course material using a drag-and-drop interface with live processing progress through **Server-Sent Events (SSE)**.

### 🔬 Document Inspection Studio

Inspect processed documents through a split-pane interface:

* Original PDF page scans
* Extracted Markdown
* LaTeX formulas rendered with KaTeX
* Inline editing capabilities

### 🖥️ Hardware & VRAM Monitoring

Monitor local hardware resources and GPU/VRAM health from the dashboard.

---

# 🚀 Quickstart

## Prerequisites

Make sure the following are installed:

* Python 3.x
* Docker
* Docker Compose
* A Discord application/bot
* The required local AI model runtime

---

## 1. Start Qdrant

Start the Qdrant vector database using Docker Compose:

```bash
docker compose up -d
```

Qdrant will be available at:

```text
http://localhost:6333
```

The Qdrant dashboard is available at:

```text
http://localhost:6333/dashboard
```

---

## 2. Configure Environment Variables

Create your local environment configuration from the example file:

### Windows

```powershell
copy .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

Edit `.env` and configure the required services:

```ini
DISCORD_BOT_TOKEN="your_token_here"

OLLAMA_BASE_URL="http://localhost:11434"

QDRANT_HOST="localhost"
QDRANT_PORT=6333
```

> **Note:** Do not commit your `.env` file or Discord bot token to version control.

---

## 3. Launch the Web Dashboard

Start the FastAPI application using Uvicorn:

```bash
uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://localhost:8000
```

The dashboard can then be used to manage subjects, upload course material, inspect processed documents, and monitor system resources.

---

## 4. Launch the Discord Bot

Start the Discord bot with:

```bash
python -m src.bot.bot
```

Once the bot is connected to Discord, it can be used through the configured server and channels.

---

# 💬 Discord Commands

| Command                        | Description                                                                 |
| ------------------------------ | --------------------------------------------------------------------------- |
| `/bind <subject_id>`           | Bind the current text channel to a subject. Example: `/bind calculus_1`     |
| `/unbind`                      | Remove the subject binding from the current channel.                        |
| `/status`                      | Display service health, VRAM usage, and vector count for the bound subject. |
| `/subjects`                    | List registered academic subjects and their associated channels.            |
| `/ask <question> [subject_id]` | Ask CampusAI a question, optionally overriding the channel's bound subject. |
| `/latex <code> [scale] [theme]` | Render a mathematical TeX expression without asking the LLM. |

### Mathematical answers in Discord

Install Node.js 24 LTS (verified with 24.21.0) on the machine running the bot,
then install the locked local renderer dependencies:

```powershell
npm ci --prefix tools/latex-renderer
```

The bot checks the renderer at startup. When available, subject-channel replies
and `/ask` can show equations automatically as labeled PNG attachments alongside
their explanations and citations. The model chooses when notation is useful;
ordinary answers stay as text. Use `/latex` for a direct render of mathematical
TeX such as `\frac{a}{b}`, `\begin{pmatrix}1&2\\3&4\end{pmatrix}`, or an
integral. Bare TeX and one enclosing `$$...$$`, `\[...\]`, or `\(...\)` pair
are accepted. Choose a light or dark image background and a scale from 0.5 to 3.

The helper works locally without a browser or TeX compiler. It handles math
commands from MathJax's base and AMS packages, with up to four expressions per
answer and 2,000 characters per expression. Full LaTeX documents, arbitrary
packages, and resource loading commands are unsupported. The `latex` section in
`config.yaml` controls availability, theme, scale, timeout, and size limits. If
startup reports the renderer unavailable, check Node on `PATH`, run the `npm ci`
command above on that host, and restart the bot. Text answers remain available
when the renderer or Attach Files permission is unavailable. Install packages on
each deployment platform; do not copy `node_modules` between Windows and Linux.

Rendering uses one local process per batch, with a 10-second deadline and no
waiting queue. Busy, invalid, oversized, or failed equations fall back to copyable
TeX without regenerating the answer. Each image is limited to 2,048 × 1,024 pixels
and 1 MiB; smaller destination upload limits are also respected. A process failure
disables rendering until a successful probe or restart. The bot never installs
dependencies during startup. Set `latex.node_executable` to an absolute Node path
if the bot's service account has a different `PATH`.

Run the Python checks without Node using `python -m pytest tests/ -m "not latex_real"`.
After `npm ci`, run `python -m pytest tests/` to include real rendering. The real
renderer checks are marked `latex_real` and skip if Node or the local dependencies
are absent. Set `LATEX_TEST_NODE` to a particular Node executable when checking
another runtime. Fixtures cover both image themes, integrals, matrices, aligned
rows, cases, invalid TeX, parser isolation, and bounded process cleanup. Live
Discord desktop/mobile display and permissions still need a server smoke test.

### Example

```text
/ bind calculus_1
```

The channel is now associated with the `calculus_1` subject.

Users can then ask questions normally through the bot or explicitly use:

```text
/ask What is the fundamental theorem of calculus?
```

---

# 📚 RAG Pipeline

Course material follows a retrieval pipeline before being provided to the LLM.

```text
PDF / Course Material
        │
        ▼
  GLM-OCR + Gemma vision
        │
        ▼
Document Processing
        │
        ▼
    BGE-M3
   Embeddings
        │
        ▼
     Qdrant
   Vector Storage
        │
        ▼
 Semantic Retrieval
        │
        ▼
BGE Reranker v2 M3
        │
        ▼
 Relevant Context
        │
        ▼
gemma4_e2b_q8:latest
        │
        ▼
    Final Answer
```

This allows the bot to answer questions using the specific course material associated with a Discord channel.

---

# 🧪 Running the Test Suite

PDF ingestion uses the PDF text layer by default. Pages with fewer than 80
non-whitespace characters are sent to the configured OCR model through
Ollama. Keep Ollama running during ingestion.
Actual page rendering requires Poppler (`pdfinfo` and `pdftoppm` on PATH);
generated text previews are never used as OCR or vision input.

The `ingestion` settings in `config.yaml` control this behavior:

* `ocr_min_text_chars: 80`: minimum usable text length before OCR is requested.
* `ocr_formula_pages: false`: enable to also OCR pages containing math-heavy
  lines. This is a heuristic; it cannot detect every missing image-based formula.
* `ocr_keep_alive: "0"`: unload OCR after each request to leave room on an 8 GB GPU.
* `vision_enabled: true`: inspect pages with PDF images, form objects, painted
  vector graphics, or little extracted text. This is a candidate heuristic;
  Gemma returns a structured assessment and skips prose-only pages and decoration.
* `vision_model: "gemma4_e2b_q8:latest"`: use the same Gemma model for chat and vision.
* `vision_max_tokens: 1024`: bound visual descriptions; truncated output is rejected.

Gemma requires both the text GGUF and its matching multimodal projector. Check
`ollama show gemma4_e2b_q8:latest` for the `vision` capability before ingestion.
A text-only GGUF import will reject images even though the Gemma family is multimodal.
Chat and vision share an 8K context (`ollama.llm_num_ctx`) and disable thinking.
Router, embeddings and reranker remain on CPU. Router and embedding requests use
`keep_alive: "0"` to release RAM after each call on this 16 GB machine; this trades
some model loading time for lower idle memory use. Gemma remains loaded for chat.

OCR output is saved directly as page Markdown and then chunked and embedded.
If OCR fails, available PDF text is retained with a progress warning. If a page
has no extracted text and OCR fails, ingestion stops before indexing. An empty
OCR result can still be indexed if Gemma recovers a meaningful image or diagram
description; otherwise ingestion stops, including for a completely blank page. Results
include `ocr_pages`, `vision_pages` and `warnings`. Visual descriptions are appended
as AI-generated interpretation to page Markdown and indexed with the page's source
metadata. Vision failures retain available text and report a warning.
Re-upload previously parsed PDFs to apply OCR and visual interpretation;
existing documents are not automatically reprocessed.

OCR requests use an 8K context and a 4K output limit. Truncated responses are
rejected rather than indexed as complete text.

Run the complete test suite with:

```bash
pytest tests/
```

For more detailed output:

```bash
pytest tests/ -v
```

---

# 📁 Project Structure

A simplified overview of the project structure:

```text
.
├── src/
│   ├── bot/
│   │   └── ...
│   │
│   ├── web/
│   │   └── ...
│   │
│   └── ...
│
├── tests/
│   └── ...
│
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

| Layer                   | Technology              |
| ----------------------- | ----------------------- |
| **Discord Integration** | Discord Bot API         |
| **Backend**             | Python + FastAPI        |
| **ASGI Server**         | Uvicorn                 |
| **Frontend**            | HTML + CSS + JavaScript |
| **Vector Database**     | Qdrant                  |
| **LLM**                 | gemma4_e2b_q8:latest      |
| **Vision**              | gemma4_e2b_q8:latest     |
| **OCR**                 | GLM-OCR (`glm-ocr-eot`)  |
| **Embeddings**          | BGE-M3                  |
| **Reranking**           | BGE Reranker v2 M3      |
| **Intent Router**       | Arch-Router             |
| **Containerization**    | Docker + Docker Compose |
| **Testing**             | Pytest                  |

---

# 🔐 Local-First AI

One of the primary goals of CampusAI is to keep the AI pipeline local.

Instead of sending course material and questions to a third-party hosted AI service, the application is designed around locally hosted models and infrastructure.

This provides greater control over:

* 📄 Course material
* 🔒 Data privacy
* 🧠 Model selection
* ⚡ Inference configuration
* 🖥️ Hardware utilization
* 🔧 System customization

---

# 📄 License

This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
