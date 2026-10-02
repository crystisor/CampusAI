# 🎓 CampusAI — Discord Bot Powered by a Local AI Stack

A Discord-based academic assistant that answers course-related questions using a **fully local AI stack**.

CampusAI combines **LLM inference, RAG, document processing, web search, reranking, and intent routing** to provide context-aware answers directly inside Discord.

The system is designed around privacy, local inference, and efficient retrieval — allowing academic course material to remain on the local machine.

---

## ✨ Features

* 🤖 **Discord AI Assistant** — Ask questions directly from Discord.
* 🧠 **Local LLM inference** — Powered by `Spark-X2.5-4b-Q8_0`.
* 📚 **RAG-based course knowledge** — Course material is indexed in Qdrant.
* 🔎 **Semantic search** — Uses `bge-m3` embeddings.
* 🎯 **Reranking** — Uses `bge-reranker-v2-m3` to select the most relevant context.
* 🌐 **Web search** — Supports questions requiring information outside the indexed course material.
* 🧭 **Intent routing** — `Arch-Router` determines how each query should be handled.
* 👁️ **Visual document processing** — `glm-ocr` processes visual content and documents.
* 📄 **PDF ingestion** — Upload and process course material through the web dashboard.
* 🖥️ **Management dashboard** — Monitor subjects, documents, hardware, VRAM, and vector database status.
* 🔒 **Local-first architecture** — AI inference and vector storage can run entirely on your own hardware.

---

# 🏗️ Architecture

CampusAI uses a modular architecture consisting of:

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
| **LLM**             | `Spark-X2.5-4b-Q8_0` | Answer generation                     |
| **Visual Model**    | `glm-ocr`            | OCR and visual document understanding |
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
                         Spark-X2.5-4b-Q8_0
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
     GLM-OCR
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
Spark-X2.5-4b-Q8_0
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
generated text previews are never used as OCR input.

The `ingestion` settings in `config.yaml` control this behavior:

* `ocr_min_text_chars: 80`: minimum usable text length before OCR is requested.
* `ocr_formula_pages: false`: enable to also OCR pages containing math-heavy
  lines. This is a heuristic; it cannot detect every missing image-based formula.
* `ocr_keep_alive: "1m"`: keep the OCR model loaded for one minute after a request.

OCR output is saved directly as page Markdown and then chunked and embedded.
If OCR fails, available PDF text is retained with a progress warning. If a page
has no extracted text and OCR fails or returns empty output, ingestion stops
before indexing. This can also happen for a completely blank page. Results
include `ocr_pages` and `warnings`. Re-upload previously parsed PDFs to apply OCR;
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
| **LLM**                 | Spark-X2.5-4b-Q8_0      |
| **Vision / OCR**        | GLM-OCR                 |
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
