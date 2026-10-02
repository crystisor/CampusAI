from src.bot.cogs.study_chat import source_formula_page
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
