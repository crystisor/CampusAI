import re
from typing import List, Dict, Any

class MarkdownFormulaChunker:
    """
    Splits Markdown files into chunks while preserving math equations ($$...$$ and $...$).
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_page_markdown(
        self,
        markdown_text: str,
        subject_id: str,
        document_name: str,
        page_number: int,
    ) -> List[Dict[str, Any]]:
        """
        Chunks the page markdown and adds metadata.
        """
        # Protect math blocks from being split
        # Replace display math $$...$$ with placeholders
        math_blocks = []
        def _math_replacer(match):
            placeholder = f"__MATH_BLOCK_{len(math_blocks)}__"
            math_blocks.append(match.group(0))
            return placeholder

        protected_text = re.sub(r"\$\$(.*?)\$\$", _math_replacer, markdown_text, flags=re.DOTALL)

        # Split by paragraphs or markdown headers
        paragraphs = re.split(r"\n\s*\n", protected_text)
        raw_chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            p_len = len(para)
            if current_len + p_len > self.chunk_size and current_chunk:
                raw_chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = p_len
            else:
                current_chunk.append(para)
                current_len += p_len + 2

        if current_chunk:
            raw_chunks.append("\n\n".join(current_chunk))

        # Restore math blocks and wrap into structured chunk objects
        chunks: List[Dict[str, Any]] = []
        for idx, chunk_text in enumerate(raw_chunks):
            for i, block in enumerate(math_blocks):
                chunk_text = chunk_text.replace(f"__MATH_BLOCK_{i}__", block)
            
            chunks.append({
                "chunk_id": f"{document_name}_p{page_number}_c{idx}",
                "text": chunk_text.strip(),
                "subject_id": subject_id,
                "document_name": document_name,
                "page_number": page_number,
            })

        return chunks
