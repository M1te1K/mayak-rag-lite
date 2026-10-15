from pathlib import Path

import pytest
from conftest import WordTokenizer
from pypdf import PdfWriter

from mayak.pdf import DocumentError, load_pdf, make_chunks
from mayak.search import fit_context

ROOT = Path(__file__).resolve().parents[1]


def test_real_demo_pdf_text_pages_and_stable_chunks():
    doc = load_pdf(ROOT / "output/pdf/mayak-demo.pdf")
    assert len(doc.pages) == 3
    assert "ДЕМОНСТРАЦИОННЫЙ ДОКУМЕНТ" in doc.pages[0].text
    assert "390 рублей" in doc.pages[0].text
    assert "4 рабочих часов" in doc.pages[1].text
    chunks = make_chunks(doc, WordTokenizer(), 40, 8)
    assert chunks == make_chunks(doc, WordTokenizer(), 40, 8)
    assert len({c.id for c in chunks}) == len(chunks)
    assert {c.page for c in chunks} == {1, 2, 3}
    assert all(c.text in doc.pages[c.page - 1].text for c in chunks)
    assert chunks[0].text.split()[-8:] == chunks[1].text.split()[:8]


def test_changed_document_changes_ids(tmp_path):
    from dataclasses import replace

    doc = load_pdf(ROOT / "output/pdf/mayak-demo.pdf")
    assert (
        make_chunks(doc, WordTokenizer(), 40, 8)[0].id
        != make_chunks(replace(doc, digest="changed"), WordTokenizer(), 40, 8)[0].id
    )


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("missing", "не найден"),
        ("empty", "пустой"),
        ("broken", "повреждён"),
        ("scan", "OCR"),
        ("encrypted", "зашифрован"),
    ],
)
def test_pdf_errors(tmp_path, kind, expected):
    path = tmp_path / "input.pdf"
    if kind == "empty":
        path.write_bytes(b"")
    elif kind == "broken":
        path.write_bytes(b"not a PDF")
    elif kind in {"scan", "encrypted"}:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        if kind == "encrypted":
            writer.encrypt("pass")
        writer.write(path)
    with pytest.raises(DocumentError, match=expected):
        load_pdf(path)


def test_vector_ranking_neighbors_and_budget(search):
    ranked = search.rank("Сколько стоит Старт?", 4)
    assert ranked[0].chunk.id == "c1"
    assert ranked[0].score > ranked[-1].score
    expanded = search.expand(ranked, 4)
    assert {h.chunk.id for h in expanded} >= {"c1", "c2"}
    fitted = fit_context(expanded, 110)
    assert sum(len(h.chunk.text) for h in fitted) <= 110


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_expansion_respects_limit(search, limit):
    ranked = search.rank("Старт", 4)
    expanded = search.expand(ranked, limit)
    assert len(expanded) <= limit
    assert len({h.chunk.id for h in expanded}) == len(expanded)
    assert all(abs(h.score - search.last_scores[h.chunk.id]) < 1e-6 for h in expanded)
