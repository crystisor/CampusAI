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

def test_chunk_overlap():
    chunker = MarkdownFormulaChunker(chunk_size=150, chunk_overlap=80)
    markdown_text = (
        "Paragraph One introduces the first core concept of algorithmic complexity.\n\n"
        "Paragraph Two elaborates on Big-O notation and asymptotic upper bounds in computer science.\n\n"
        "Paragraph Three discusses Big-Omega notation for lower bounds.\n\n"
        "Paragraph Four summarizes tight bounds using Big-Theta notation."
    )

    chunks = chunker.chunk_page_markdown(
        markdown_text=markdown_text,
        subject_id="algorithms",
        document_name="complexity",
        page_number=1
    )

    assert len(chunks) > 1
    # Check that trailing content from chunk 0 is carried over into chunk 1
    chunk_0_text = chunks[0]["text"]
    chunk_1_text = chunks[1]["text"]
    # Verify overlap exists between consecutive chunks
    overlap_found = any(line in chunk_1_text for line in chunk_0_text.split("\n\n") if len(line) > 20)
    assert overlap_found is True

def test_default_chunker_params():
    chunker = MarkdownFormulaChunker()
    assert chunker.chunk_size == 900
    assert chunker.chunk_overlap == 150
