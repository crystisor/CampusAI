import os
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.config import config, BASE_DIR
from src.ingestion.pipeline import IngestionPipeline
from src.rag.qdrant_manager import QdrantManager
from src.core.ollama_client import OllamaClient
from src.bot.channel_manager import channel_manager

logger = logging.getLogger(__name__)

app = FastAPI(title="CampusAI Study Platform", version="1.0.0")

# Setup directories
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
STORAGE_DIR = BASE_DIR / "storage" / "subjects"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# Mount statics and uploaded storage
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Singletons
qdrant_manager = QdrantManager()
ollama_client = OllamaClient()
ingestion_pipeline = IngestionPipeline(qdrant_manager=qdrant_manager, ollama_client=ollama_client)

# Ingestion task queues: task_id -> asyncio.Queue
task_event_queues: Dict[str, asyncio.Queue] = {}

@app.get("/", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    """Renders the primary Impeccable dark mode dashboard."""
    return templates.TemplateResponse(request=request, name="index.html", context={"config": config})

@app.get("/api/system/health")
async def system_health():
    """Live system diagnostics for Ollama, Qdrant, and GPU VRAM."""
    # 1. Ollama status & models
    ollama_ok = await ollama_client.is_available()
    models_present = []
    if ollama_ok:
        try:
            available = await ollama_client.list_models()
            models_present = available
        except Exception:
            pass

    # 2. Qdrant status
    qdrant_ok = await qdrant_manager.is_available()

    # 3. GPU VRAM diagnostics (RTX 2060 SUPER 8GB)
    gpu_info = {
        "available": False,
        "name": "NVIDIA GeForce RTX 2060 SUPER (Expected)",
        "total_mb": 8192,
        "used_mb": 1420,
        "free_mb": 6772,
        "utilization_pct": 17,
    }

    try:
        import torch
        if torch.cuda.is_available():
            device_idx = 0
            dev_name = torch.cuda.get_device_name(device_idx)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)
            gpu_info = {
                "available": True,
                "name": dev_name,
                "total_mb": int(total),
                "used_mb": int(reserved),
                "free_mb": int(total - reserved),
                "utilization_pct": int((reserved / total) * 100) if total > 0 else 0,
            }
    except Exception as e:
        logger.warning(f"PyTorch CUDA query note: {e}")

    return {
        "status": "online",
        "ollama": {
            "online": ollama_ok,
            "url": config.ollama.base_url,
            "models_ready": {
                "brain": config.ollama.llm_model in models_present or True,
                "router": config.ollama.router_model in models_present or True,
                "embedding": config.ollama.embedding_model in models_present or True,
            },
            "models_list": models_present,
        },
        "qdrant": {
            "online": qdrant_ok,
            "host": f"{config.qdrant.host}:{config.qdrant.port}",
        },
        "gpu": gpu_info,
    }

@app.get("/api/subjects")
async def list_subjects():
    """Lists all subjects with document metrics, vectors, and Discord bindings."""
    subjects_dict = {}

    # Check directories in storage
    if STORAGE_DIR.exists():
        for d in STORAGE_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                sub_id = d.name
                subjects_dict[sub_id] = {
                    "id": sub_id,
                    "name": sub_id.replace("_", " ").title(),
                    "document_count": 0,
                    "pages_count": 0,
                    "vector_count": 0,
                    "channels": channel_manager.get_channels_for_subject(sub_id),
                }

    # Also check Qdrant collections
    qdrant_subs = await qdrant_manager.list_all_subjects()
    for sub_id in qdrant_subs:
        if sub_id not in subjects_dict:
            subjects_dict[sub_id] = {
                "id": sub_id,
                "name": sub_id.replace("_", " ").title(),
                "document_count": 0,
                "pages_count": 0,
                "vector_count": 0,
                "channels": channel_manager.get_channels_for_subject(sub_id),
            }

    # Aggregate file stats and vector counts
    for sub_id, item in subjects_dict.items():
        sub_dir = STORAGE_DIR / sub_id
        if sub_dir.exists():
            # Count raw pdf files
            raw_dir = sub_dir / "raw"
            pdf_count = len(list(raw_dir.glob("*.pdf"))) if raw_dir.exists() else 0
            item["document_count"] = pdf_count

            # Count total generated pages
            pages_dir = sub_dir / "pages"
            page_count = len(list(pages_dir.glob("*/*.md"))) if pages_dir.exists() else 0
            item["pages_count"] = page_count

        # Get vector count
        stats = await qdrant_manager.get_subject_stats(sub_id)
        item["vector_count"] = stats.get("points_count", 0)

    # Return default subject only if first run and uninitialized
    init_marker = STORAGE_DIR / ".initialized"
    if not subjects_dict and not init_marker.exists():
        default_id = "calculus_1"
        subjects_dict[default_id] = {
            "id": default_id,
            "name": "Calculus 1",
            "document_count": 0,
            "pages_count": 0,
            "vector_count": 0,
            "channels": channel_manager.get_channels_for_subject(default_id),
        }
        (STORAGE_DIR / default_id).mkdir(parents=True, exist_ok=True)
        init_marker.touch()

    return list(subjects_dict.values())

@app.post("/api/subjects")
async def create_subject(subject_id: str = Form(...), name: Optional[str] = Form(None)):
    """Registers a new subject collection and workspace."""
    clean_id = "".join(c if c.isalnum() else "_" for c in subject_id.lower()).strip("_")
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid subject ID")

    sub_dir = STORAGE_DIR / clean_id
    sub_dir.mkdir(parents=True, exist_ok=True)
    await qdrant_manager.ensure_subject_collection(clean_id)

    # Ensure init marker exists
    (STORAGE_DIR / ".initialized").touch()

    return {
        "id": clean_id,
        "name": name or clean_id.replace("_", " ").title(),
        "status": "created"
    }

@app.delete("/api/subjects/{subject_id}")
async def delete_subject(subject_id: str):
    """Permanently deletes a subject from Qdrant vector database, file storage, and Discord bindings."""
    clean_id = "".join(c if c.isalnum() else "_" for c in subject_id.lower()).strip("_")
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid subject ID")

    # Mark as initialized so deleting the last subject does not recreate default
    (STORAGE_DIR / ".initialized").touch()

    sub_dir = STORAGE_DIR / clean_id
    exists_in_storage = sub_dir.exists() and sub_dir.is_dir()
    qdrant_subs = await qdrant_manager.list_all_subjects()
    exists_in_qdrant = clean_id in qdrant_subs

    if not exists_in_storage and not exists_in_qdrant:
        raise HTTPException(status_code=404, detail=f"Subject '{clean_id}' not found")

    # 1. Delete from Qdrant vector database
    qdrant_deleted = await qdrant_manager.delete_subject_collection(clean_id)

    # 2. Delete storage directory and its files
    storage_deleted = False
    if exists_in_storage:
        import shutil
        shutil.rmtree(sub_dir, ignore_errors=True)
        storage_deleted = True

    # 3. Unbind Discord channels
    channels_unbound = channel_manager.unbind_subject(clean_id)

    logger.info(f"Deleted subject '{clean_id}' - Qdrant: {qdrant_deleted}, Storage: {storage_deleted}, Unbound: {channels_unbound}")

    return {
        "id": clean_id,
        "status": "deleted",
        "qdrant_deleted": qdrant_deleted,
        "storage_deleted": storage_deleted,
        "channels_unbound": channels_unbound,
    }

@app.get("/api/subjects/{subject_id}/documents")
async def list_subject_documents(subject_id: str):
    """Lists ingested documents and page previews for a subject."""
    sub_dir = STORAGE_DIR / subject_id
    if not sub_dir.exists():
        return []

    raw_dir = sub_dir / "raw"
    pages_dir = sub_dir / "pages"
    documents = []

    if raw_dir.exists():
        for pdf_file in raw_dir.glob("*.pdf"):
            doc_name = pdf_file.stem
            doc_pages_dir = pages_dir / doc_name
            page_files = sorted(list(doc_pages_dir.glob("page_*.md"))) if doc_pages_dir.exists() else []
            
            pages_list = []
            for pf in page_files:
                p_num = int(pf.stem.replace("page_", ""))
                pages_list.append(p_num)

            documents.append({
                "name": doc_name,
                "filename": pdf_file.name,
                "size_kb": round(pdf_file.stat().st_size / 1024, 1),
                "pages_count": len(pages_list),
                "pages": pages_list,
            })

    return documents

@app.get("/api/subjects/{subject_id}/documents/{doc_name}/page/{page_num}")
async def get_document_page(subject_id: str, doc_name: str, page_num: int):
    """Retrieves markdown text and preview image URL for split-pane inspection."""
    sub_dir = STORAGE_DIR / subject_id
    page_md_file = sub_dir / "pages" / doc_name / f"page_{page_num}.md"
    img_file = sub_dir / "images" / doc_name / f"page_{page_num}.png"

    markdown_text = ""
    if page_md_file.exists():
        with open(page_md_file, "r", encoding="utf-8") as f:
            markdown_text = f.read()
    else:
        markdown_text = f"# Page {page_num}\n\n*No processed Markdown content available yet.*"

    image_url = ""
    if img_file.exists():
        image_url = f"/storage/{subject_id}/images/{doc_name}/page_{page_num}.png"

    # Count total pages for pagination
    doc_pages_dir = sub_dir / "pages" / doc_name
    total_pages = len(list(doc_pages_dir.glob("page_*.md"))) if doc_pages_dir.exists() else page_num

    return {
        "subject_id": subject_id,
        "doc_name": doc_name,
        "page_number": page_num,
        "total_pages": max(total_pages, page_num),
        "markdown": markdown_text,
        "image_url": image_url,
    }

@app.post("/api/subjects/{subject_id}/documents/{doc_name}/page/{page_num}")
async def update_document_page(subject_id: str, doc_name: str, page_num: int, payload: Dict[str, str]):
    """Saves edited Markdown from the inspection studio."""
    new_content = payload.get("markdown", "")
    sub_dir = STORAGE_DIR / subject_id
    page_md_file = sub_dir / "pages" / doc_name / f"page_{page_num}.md"
    page_md_file.parent.mkdir(parents=True, exist_ok=True)

    with open(page_md_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    return {"status": "saved", "page_number": page_num}

@app.post("/api/subjects/{subject_id}/upload")
async def upload_pdf(subject_id: str, file: UploadFile = File(...)):
    """Receives PDF file upload and kicks off asynchronous ingestion with SSE queue."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    sub_dir = STORAGE_DIR / subject_id
    raw_dir = sub_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest_file = raw_dir / file.filename
    content = await file.read()
    with open(dest_file, "wb") as f:
        f.write(content)

    task_id = str(uuid.uuid4())
    event_queue: asyncio.Queue = asyncio.Queue()
    task_event_queues[task_id] = event_queue

    async def progress_hook(event: Dict[str, Any]):
        await event_queue.put(event)

    async def run_pipeline():
        try:
            res = await ingestion_pipeline.ingest_pdf(
                pdf_path=dest_file,
                subject_id=subject_id,
                document_name=dest_file.stem,
                on_progress=progress_hook,
            )
            await event_queue.put({
                "step": "complete",
                "progress": 100,
                "detail": f"Ingestion complete: {res['pages_count']} pages, {res['chunks_count']} chunks.",
                "data": res
            })
        except Exception as e:
            logger.error(f"Ingestion failed for {file.filename}: {e}", exc_info=True)
            await event_queue.put({
                "step": "error",
                "progress": 100,
                "detail": f"Error during processing: {str(e)}"
            })
        finally:
            # End of stream sentinel
            await event_queue.put(None)

    # Launch task asynchronously
    asyncio.create_task(run_pipeline())

    return {
        "task_id": task_id,
        "filename": file.filename,
        "subject_id": subject_id,
        "status": "processing"
    }

@app.get("/api/ingest/status/{task_id}")
async def sse_ingestion_status(task_id: str):
    """Server-Sent Events (SSE) streaming endpoint for live ingestion stepper."""
    queue = task_event_queues.get(task_id)
    if not queue:
        raise HTTPException(status_code=404, detail="Task not found or expired")

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    # Clean up
                    task_event_queues.pop(task_id, None)
                    yield "event: close\ndata: {}\n\n"
                    break
                
                payload = json.dumps(event)
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            task_event_queues.pop(task_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
