# CampusAI — College Study Companion & Local AI Stack

An offline-capable, privacy-preserving academic study companion for college students, featuring:
- **Discord Bot**: Intent-routed study assistant with multi-channel subject bindings, cross-encoder reranking, LaTeX math answers, and source citations.
- **Impeccable Web Dashboard**: Deep Slate / Obsidian dark mode dashboard (`#090D16` / `#0F172A`) for PDF document ingestion, live SSE stepper (`pdf_split` ➔ `layout` ➔ `ocr` ➔ `markdown` ➔ `embed` ➔ `indexed`), and a split-pane Document & Formula Inspection Studio.
- **Formula-Preserving OCR Pipeline**: `pp-doclayoutV3` layout detection, `glm-ocr` formula and text extraction, and math-preserving markdown chunking.
- **Local Model Stack**: Ollama (`Spark-X2.5-4b-Q8_0`, `Arch-Router`, `bge-m3`), Qdrant Vector Database, and `bge-reranker-v2-m3`.

---

## Architecture Overview

```
Discord Channel (#calculus-1)
       │
       ▼
Arch-Router (Intent Classification)
       ├── DIRECT  ───────────────► Spark-X2.5-4b-Q8_0 ──► Discord Reply
       ├── RAG     ─► Qdrant (bge-m3) ─┐
       ├── WEB     ─► Web Search (DDG) ┼─► bge-reranker-v2-m3 (Top 4-5) ─► Spark-X2.5-4b-Q8_0
       └── HYBRID  ─► Qdrant + Web   ──┘
```

---

## Quickstart

### 1. Start Qdrant Vector Database
```bash
docker compose up -d
```
Qdrant will be available at `http://localhost:6333` with web dashboard at `http://localhost:6333/dashboard`.

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your Discord Bot Token:
```bash
copy .env.example .env
```
Edit `.env`:
```ini
DISCORD_BOT_TOKEN="your_token_here"
OLLAMA_BASE_URL="http://localhost:11434"
QDRANT_HOST="localhost"
QDRANT_PORT=6333
```

### 3. Launch Impeccable Web Management Dashboard
```bash
uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://localhost:8000` to access:
- **Subject Directory & Channel Badges**
- **PDF Upload & Drag-and-Drop Zone** with live SSE horizontal stepper
- **Split-Pane Inspection Studio**: Original PDF page scans on the left, KaTeX-rendered LaTeX formulas and Markdown on the right, with live inline editor toggle.
- **Hardware & VRAM Health Drawer**

### 4. Launch the Discord Bot
```bash
python -m src.bot.bot
```

---

## Discord Bot Commands

| Command | Description |
| :--- | :--- |
| `/bind <subject_id>` | Binds the current text channel to a subject (e.g. `/bind calculus_1`) |
| `/unbind` | Removes subject binding from the current channel |
| `/status` | Shows service health, VRAM, and vector count for the bound subject |
| `/subjects` | Lists all registered college subjects and their bound channels |
| `/ask <question> [subject_id]` | Directly query CampusAI with optional subject override |

---

## Running Test Suite

```bash
pytest tests/
```
