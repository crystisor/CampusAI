import re
from typing import List, Dict, Any

class MarkdownFormulaChunker:
    """
    Splits Markdown files into chunks while preserving math equations ($$...$$ and $...$).
    """

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
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

                # Carry over trailing content from previous chunk according to chunk_overlap
                overlap_paras: List[str] = []
                overlap_len = 0
                if self.chunk_overlap > 0:
                    for prev_para in reversed(current_chunk):
                        para_cost = len(prev_para) + (2 if overlap_paras else 0)
                        if overlap_len + para_cost <= self.chunk_overlap:
                            overlap_paras.insert(0, prev_para)
                            overlap_len += para_cost
                        elif not overlap_paras:
                            # Try to take sentences from the tail of prev_para if single paragraph exceeds overlap
                            sentences = re.split(r"(?<=[.!?])\s+", prev_para)
                            tail_sentences: List[str] = []
                            tail_len = 0
                            for sent in reversed(sentences):
                                sent_cost = len(sent) + (1 if tail_sentences else 0)
                                if tail_len + sent_cost <= self.chunk_overlap:
                                    tail_sentences.insert(0, sent)
                                    tail_len += sent_cost
                                else:
                                    break
                            if tail_sentences:
                                joined_tail = " ".join(tail_sentences)
                                if "__MATH_BLOCK_" not in joined_tail or joined_tail.count("__MATH_BLOCK_") == len(re.findall(r"__MATH_BLOCK_\d+__", joined_tail)):
                                    overlap_paras.insert(0, joined_tail)
                                    overlap_len += len(joined_tail)
                            break
                        else:
                            break

                current_chunk = overlap_paras + [para]
                current_len = sum(len(p) for p in current_chunk) + 2 * max(0, len(current_chunk) - 1)
            else:
                current_chunk.append(para)
                current_len += p_len + (2 if len(current_chunk) > 1 else 0)

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
