import pytest
from src.rag.chunker import MarkdownFormulaChunker

def test_formula_preservation():
    chunker = MarkdownFormulaChunker(chunk_size=100, chunk_overlap=20)
    markdown_text = (
        "# Calculus Lecture Notes\n\n"
        "Here is the fundamental definition of a definite integral:\n\n"
        "$$\\int_{a}^{b} f(x) dx = \\lim_{n \\to \\infty} \\sum_{i=1}^{n} f(x_i^*) \\Delta x$$\n\n"
        "Notice how the Riemann sum approximates the area under the curve."
    )

    chunks = chunker.chunk_page_markdown(
        markdown_text=markdown_text,
        subject_id="calculus_1",
        document_name="derivatives",
        page_number=1
    )

    assert len(chunks) >= 1
    # Ensure the math formula was not broken
    found_formula = any("\\int_{a}^{b} f(x) dx" in c["text"] for c in chunks)
    assert found_formula is True
    assert chunks[0]["subject_id"] == "calculus_1"
    assert chunks[0]["page_number"] == 1
