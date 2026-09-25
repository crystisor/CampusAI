import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import pypdf
from src.config import config

logger = logging.getLogger(__name__)

class PDFPageData:
    def __init__(self, page_number: int, text: str, image_path: Optional[Path] = None):
        self.page_number = page_number
        self.text = text
        self.image_path = image_path

class PDFProcessor:
    """
    Handles PDF splitting, text extraction, and page image generation for visual preview and OCR.
    """

    def __init__(self, dpi: Optional[int] = None):
        self.dpi = config.ingestion.dpi if dpi is None else dpi

    def extract_pages(self, pdf_path: Path, output_image_dir: Optional[Path] = None) -> List[PDFPageData]:
        """
        Extracts raw text and generates page images from a PDF file.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        reader = pypdf.PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        pages_data: List[PDFPageData] = []

        if output_image_dir:
            output_image_dir.mkdir(parents=True, exist_ok=True)

        # Attempt high-res rendering with pdf2image if poppler is present
        pdf_images: List[Optional[Image.Image]] = [None] * num_pages
        try:
            from pdf2image import convert_from_path
            converted = convert_from_path(str(pdf_path), dpi=self.dpi)
            for i, img in enumerate(converted):
                if i < num_pages:
                    pdf_images[i] = img
        except Exception as e:
            logger.info(f"pdf2image fallback (poppler not in PATH or unavailable: {e}). Generating visual page representations.")

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            text = page.extract_text() or ""
            
            image_path: Optional[Path] = None
            if output_image_dir:
                img_file = output_image_dir / f"page_{page_num}.png"
                
                if pdf_images[idx] is not None:
                    pdf_images[idx].save(img_file, "PNG")
                    image_path = img_file
                else:
                    # Generate a clean high-contrast text render page representation
                    img = self._create_page_preview_image(text, page_num)
                    img.save(img_file, "PNG")
                    image_path = img_file

            pages_data.append(PDFPageData(page_number=page_num, text=text, image_path=image_path))

        return pages_data

    def _create_page_preview_image(self, text: str, page_num: int, width: int = 800, height: int = 1100) -> Image.Image:
        """
        Creates a crisp preview canvas of the document page when poppler is absent.
        """
        img = Image.new("RGB", (width, height), color="#FFFFFF")
        draw = ImageDraw.Draw(img)

        # Page Header Banner
        draw.rectangle([(0, 0), (width, 50)], fill="#0F172A")
        draw.text((25, 16), f"PAGE {page_num} — DOCUMENT PREVIEW", fill="#38BDF8")

        # Page content lines
        lines = text.split("\n")[:40]
        y_offset = 75
        for line in lines:
            line_str = line.strip()
            if not line_str:
                y_offset += 16
                continue
            
            # Format display
            draw.text((35, y_offset), line_str[:90], fill="#1E293B")
            y_offset += 24
            if y_offset > height - 60:
                break

        # Page Footer
        draw.line([(30, height - 40), (width - 30, height - 40)], fill="#CBD5E1", width=1)
        draw.text((width - 120, height - 30), f"Page {page_num}", fill="#64748B")

        return img
