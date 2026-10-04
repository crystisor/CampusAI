from src.bot.cogs.study_chat import format_references, source_formula_page
from src.config import config


def test_formula_request_uses_rendered_page_from_retrieved_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    subject = tmp_path / "articles"
    pdf = subject / "raw" / "Paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4")
    contexts = [{"source": "course_material", "metadata": {
        "document_name": "Paper", "page_number": 5,
    }}]

    assert source_formula_page("Explain and print the QLTY formula", contexts, "articles") == (pdf, 5)
    assert source_formula_page("Summarize this paper", contexts, "articles") is None
    pdf.unlink()
    assert source_formula_page("Explain the formula", contexts, "articles") is None


def test_formula_page_rejects_metadata_outside_subject(tmp_path, monkeypatch):
    monkeypatch.setattr(config.ingestion, "storage_dir", tmp_path)
    contexts = [{"source": "course_material", "metadata": {
        "document_name": "../../outside", "page_number": 5,
    }}]
    assert source_formula_page("Print the formula", contexts, "articles") is None


def test_references_include_every_course_page_and_web_source():
    contexts = [
        {"source": "course_material", "metadata": {"document_name": "Lecture 2", "page_number": 4}},
        {"source": "course_material", "metadata": {"document_name": "Lecture 2", "page_number": 4}},
        {"source": "course_material", "metadata": {"document_name": "Lecture 2", "page_number": 7}},
        {"source": "course_material", "metadata": {"document_name": "Lecture 3", "course_number": 3, "page_number": 2}},
        {"source": "web_search", "metadata": {"url": "https://example.org/source"}},
    ]

    references = format_references(contexts)

    assert "Course: Lecture 2 — Slides/pages 4, 7" in references
    assert "Course 3 (Lecture 3) — Slide/page 2" in references
    assert "Web: https://example.org/source" in references
    assert references.count("Lecture 2 —") == 1
    assert format_references([]) == ""
