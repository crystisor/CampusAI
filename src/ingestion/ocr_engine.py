import re
from typing import Optional

from src.config import config
from src.core.ollama_client import OllamaClient
from src.ingestion.pdf_processor import PDFPageData


class GLMOCREngine:
    """Select pages for OCR and recognize their actual rendered image via Ollama."""

    def __init__(self, model_name: Optional[str] = None,
                 ollama_client: Optional[OllamaClient] = None):
        self.model_name = model_name or config.ingestion.ocr_model
        self.ollama = ollama_client or OllamaClient()

    def ocr_reason(self, text: str) -> Optional[str]:
        if not text.strip() or len("".join(text.split())) < config.ingestion.ocr_min_text_chars:
            return "little or no extracted text"
        if config.ingestion.ocr_formula_pages:
            for line in text.splitlines():
                chars = "".join(line.split())
                # Digits alone (dates, page numbers) do not imply a formula.
                if len(chars) > 4 and any(c in "=+^_∑∫√≤≥" for c in chars):
                    density = sum(c.isdigit() or c in "=+-*/^_{}()<>∑∫√≤≥" for c in chars) / len(chars)
                    if density > config.ingestion.math_density_threshold:
                        return "formula-heavy text (optional heuristic)"
        return None

    async def recognize_page(self, page: PDFPageData) -> str:
        if not page.image_is_rendered or not page.image_path or not page.image_path.exists():
            raise ValueError("OCR requires a real PDF page render; install Poppler and ensure it is on PATH")
        return await self.ollama.recognize_image(
            page.image_path, model=self.model_name,
            keep_alive=config.ingestion.ocr_keep_alive,
        )

    def process_page_text(self, page_text: str) -> str:
        """
        Formats extracted page text into clean Markdown with LaTeX math notation.
        Identifies inline expressions and display equations.
        """
        if not page_text:
            return ""

        # Normalize multiple line breaks
        text = re.sub(r"\r\n", "\n", page_text)
        
        # Detect common math patterns and wrap with LaTeX formatting if needed
        # e.g., variable ^ exponent, f(x) = ..., \int, \sum
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append("")
                continue

            # Detect display formulas (lines consisting mostly of math symbols)
            math_symbols = set(r"=+\-*/^_{}\[\]()<>~∑∫√πθλσ")
            char_count = len(stripped)
            math_count = sum(1 for c in stripped if c in math_symbols or c.isdigit())
            
            if char_count > 4 and (math_count / char_count) > config.ingestion.math_density_threshold and not stripped.startswith("#"):
                # Treat as standalone math formula
                if not stripped.startswith("$$"):
                    processed_lines.append(f"$$\n{stripped}\n$$")
                else:
                    processed_lines.append(stripped)
            else:
                # Detect inline equations like x_1, a^2 + b^2 = c^2
                # Convert simple superscript / subscript notation into inline LaTeX $...$
                line_inline = re.sub(r'\b([a-zA-Z])\^([0-9a-zA-Z]+)\b', r'$\1^{\2}$', line)
                line_inline = re.sub(r'\b([a-zA-Z])_([0-9a-zA-Z]+)\b', r'$\1_{\2}$', line_inline)
                processed_lines.append(line_inline)

        return "\n".join(processed_lines)
