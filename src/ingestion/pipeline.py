import os
import asyncio
import logging
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Awaitable

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
    PDF text extraction -> selective GLM-OCR -> Markdown -> embeddings -> index.
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
        self.ollama = ollama_client or OllamaClient()
        self.ocr_engine = ocr_engine or GLMOCREngine(ollama_client=self.ollama)
        self.chunker = chunker or MarkdownFormulaChunker(
            chunk_size=config.ingestion.chunk_size,
            chunk_overlap=config.ingestion.chunk_overlap
        )
        self.qdrant = qdrant_manager or QdrantManager()

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
        pages_data: List[PDFPageData] = await asyncio.to_thread(
            self.pdf_processor.extract_pages, pdf_path, output_image_dir=images_dir,
        )
        num_pages = len(pages_data)
        await emit("pdf_split", 25, f"Extracted {num_pages} pages.")

        # Select OCR per page; placeholder layout regions are not OCR evidence.
        await emit("layout", 35, f"Checking text extraction across {num_pages} pages...")
        all_page_markdowns: List[str] = []
        ocr_pages: List[int] = []
        warnings: List[str] = []

        for p in pages_data:
            reason = self.ocr_engine.ocr_reason(p.text)
            processed_page_md = self.ocr_engine.process_page_text(p.text)
            if reason:
                await emit("ocr", 35 + int(25 * (p.page_number - 1) / max(num_pages, 1)),
                           f"Reading page {p.page_number} with GLM-OCR: {reason}.")
                try:
                    # Preserve OCR Markdown/LaTeX without applying text-layer regexes.
                    processed_page_md = await self.ocr_engine.recognize_page(p)
                    ocr_pages.append(p.page_number)
                except Exception as exc:
                    warning = f"Page {p.page_number}: OCR failed ({exc})."
                    if not p.text.strip():
                        raise RuntimeError(warning + " No extracted text is available; ingestion stopped.") from exc
                    warning += " Kept the PDF text layer."
                    warnings.append(warning)
                    logger.warning(warning)
                    await emit("ocr", 35, warning)

            page_md_content = f"# {doc_name} — Page {p.page_number}\n\n{processed_page_md}\n"
            
            # Save individual page .md
            page_file = pages_md_dir / f"page_{p.page_number}.md"
            with open(page_file, "w", encoding="utf-8") as f:
                f.write(page_md_content)

            all_page_markdowns.append(page_md_content)

        await emit("ocr", 60, f"Processed {num_pages} pages; {len(ocr_pages)} used GLM-OCR, {len(warnings)} warnings.")

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
        await emit("embed", 85, f"Generating bge-m3 embeddings for {len(all_chunks)} chunks on CPU...")
        vectors: List[List[float]] = []
        batch_size = config.embeddings.batch_size
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            batch_texts = [c["text"] for c in batch]
            try:
                batch_vecs = await self.ollama.get_embeddings(
                    batch_texts,
                    model=config.ollama.embedding_model,
                    num_gpu=config.ollama.embedding_num_gpu,
                    keep_alive=config.ollama.embedding_keep_alive,
                )
                if len(batch_vecs) != len(batch):
                    raise ValueError("Embedding count does not match the input batch")
                for vector in batch_vecs:
                    if (len(vector) != config.embeddings.vector_size
                            or not all(math.isfinite(value) for value in vector)
                            or not any(value != 0 for value in vector)):
                        raise ValueError("Embedding must be finite, nonzero, and have the configured dimension")
                vectors.extend(batch_vecs)
            except Exception as e:
                raise RuntimeError(
                    f"Embedding failed for chunks {i}:{i + len(batch)}; indexing stopped."
                ) from e

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
            "ocr_pages": ocr_pages,
            "warnings": warnings,
        }
