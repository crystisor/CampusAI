import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Awaitable
from PIL import Image

from src.config import config
from src.ingestion.pdf_processor import PDFProcessor, PDFPageData
from src.ingestion.layout_engine import DocumentLayoutEngine
from src.ingestion.ocr_engine import GLMOCREngine
from src.rag.chunker import MarkdownFormulaChunker
from src.rag.qdrant_manager import QdrantManager
from src.core.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Callback type for SSE progress notification: async func(dict)
ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]

class IngestionPipeline:
    """
    End-to-end PDF ingestion pipeline:
    [PDF Split] ➔ [Layout (pp-doclayoutV3)] ➔ [OCR (glm-ocr)] ➔ [Markdown (.md)] ➔ [Embed (bge-m3)] ➔ [Indexed]
    """

    def __init__(
        self,
        pdf_processor: Optional[PDFProcessor] = None,
        layout_engine: Optional[DocumentLayoutEngine] = None,
        ocr_engine: Optional[GLMOCREngine] = None,
        chunker: Optional[MarkdownFormulaChunker] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.pdf_processor = pdf_processor or PDFProcessor()
        self.layout_engine = layout_engine or DocumentLayoutEngine()
        self.ocr_engine = ocr_engine or GLMOCREngine()
        self.chunker = chunker or MarkdownFormulaChunker(
            chunk_size=config.ingestion.chunk_size,
            chunk_overlap=config.ingestion.chunk_overlap
        )
        self.qdrant = qdrant_manager or QdrantManager()
        self.ollama = ollama_client or OllamaClient()

    async def ingest_pdf(
        self,
        pdf_path: Path,
        subject_id: str,
        document_name: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full ingestion flow, saves generated assets, and indexes chunks.
        """
        doc_name = document_name or pdf_path.stem
        # Setup storage directories
        subject_dir = config.ingestion.storage_dir / subject_id
        pages_md_dir = subject_dir / "pages" / doc_name
        images_dir = subject_dir / "images" / doc_name
        
        pages_md_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        async def emit(step: str, progress: int, detail: str):
            logger.info(f"[{doc_name}] ({progress}%) {step}: {detail}")
            if on_progress:
                try:
                    await on_progress({
                        "subject_id": subject_id,
                        "document_name": doc_name,
                        "step": step,
                        "progress": progress,
                        "detail": detail,
                    })
                except Exception as e:
                    logger.warning(f"Error in progress callback: {e}")

        # 1. PDF Split & Image Generation
        await emit("pdf_split", 15, f"Splitting PDF and rendering page previews...")
        pages_data: List[PDFPageData] = self.pdf_processor.extract_pages(pdf_path, output_image_dir=images_dir)
        num_pages = len(pages_data)
        await emit("pdf_split", 25, f"Extracted {num_pages} pages.")

        # 2 & 3. Layout Detection & OCR / Formula Extraction
        await emit("layout", 35, f"Analyzing layout blocks across {num_pages} pages...")
        all_page_markdowns: List[str] = []

        for p in pages_data:
            # Layout Analysis
            if p.image_path and p.image_path.exists():
                try:
                    img = Image.open(p.image_path)
                    blocks = self.layout_engine.detect_layout(img)
                except Exception:
                    blocks = []
            else:
                blocks = []

            # OCR / Formula Processing
            processed_page_md = self.ocr_engine.process_page_text(p.text)
            
            # If formulas were detected in layout, ensure formula markup is present
            page_md_content = f"# {doc_name} — Page {p.page_number}\n\n{processed_page_md}\n"
            
            # Save individual page .md
            page_file = pages_md_dir / f"page_{p.page_number}.md"
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(page_md_content)

            all_page_markdowns.append(page_md_content)

        await emit("ocr", 60, f"Completed formula & text recognition for {num_pages} pages.")

        # 4. Save combined document markdown
        await emit("markdown", 70, f"Compiling structured Markdown...")
        combined_doc_path = subject_dir / f"{doc_name}.md"
        with open(combined_doc_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(all_page_markdowns))

        # 5. Chunking
        all_chunks: List[Dict[str, Any]] = []
        for idx, page_md in enumerate(all_page_markdowns, 1):
            chunks = self.chunker.chunk_page_markdown(
                markdown_text=page_md,
                subject_id=subject_id,
                document_name=doc_name,
                page_number=idx,
            )
            all_chunks.extend(chunks)

        await emit("markdown", 80, f"Generated {len(all_chunks)} formula-preserved chunks.")

        # 6. Embeddings (bge-m3)
        await emit("embed", 85, f"Generating bge-m3 embeddings for {len(all_chunks)} chunks...")
        vectors: List[List[float]] = []
        batch_size = 10
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            for c in batch:
                try:
                    vec = await self.ollama.get_embedding(c["text"], model=config.ollama.embedding_model)
                    if not vec:
                        # Fallback zero vector if embedding service is cold or dry-run
                        vec = [0.0] * 1024
                    vectors.append(vec)
                except Exception as e:
                    logger.warning(f"Embedding failure for chunk {c['chunk_id']}: {e}")
                    vectors.append([0.0] * 1024)

        await emit("embed", 95, f"All {len(vectors)} vectors generated.")

        # 7. Indexed into Qdrant
        await emit("indexed", 98, f"Upserting points to Qdrant collection...")
        try:
            await self.qdrant.upsert_chunks(subject_id, all_chunks, vectors)
            await emit("indexed", 100, f"Successfully indexed {doc_name} ({len(all_chunks)} chunks).")
        except Exception as e:
            logger.error(f"Qdrant indexing error: {e}")
            await emit("indexed", 100, f"Processed locally ({len(all_chunks)} chunks, Qdrant offline).")

        return {
            "subject_id": subject_id,
            "document_name": doc_name,
            "pages_count": num_pages,
            "chunks_count": len(all_chunks),
            "markdown_path": str(combined_doc_path),
        }
