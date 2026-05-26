"""Tests for ragdx.loaders.pdf.

Uses a real PDF fixture (created on the fly via reportlab if available,
else skipped). Kept tiny so it's fast.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from ragdx.loaders.pdf import PDFLoaderResult, _normalise, load_pdf_chunks


def _tiny_pdf_bytes() -> bytes | None:
    """Synthesise a minimal PDF on the fly. ``None`` if reportlab missing."""
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except Exception:
        return None
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    # Two pages, each with a paragraph long enough to chunk meaningfully.
    c.drawString(72, 720, "ASMPT 2023 ESG Report")
    c.drawString(72, 700, "Page 1 talks about climate disclosure aligned with TCFD.")
    c.drawString(72, 680, "Net-zero commitments and scope 1/2/3 emissions reporting.")
    c.showPage()
    c.drawString(72, 720, "Supply chain due diligence and conflict minerals.")
    c.drawString(72, 700, "Worker safety, training hours, gender diversity statistics.")
    c.drawString(72, 680, "Board composition and ESG-linked executive compensation.")
    c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    payload = _tiny_pdf_bytes()
    if payload is None:
        pytest.skip("reportlab not installed; skipping PDF roundtrip tests")
    path = tmp_path / "tiny.pdf"
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------- utils
def test_normalise_collapses_whitespace_and_strips_control_chars():
    raw = "Line  one\n\n  Line\ttwo  \r\n"
    cleaned = _normalise(raw)
    assert "  " not in cleaned
    assert cleaned.startswith("Line one")
    assert "" not in cleaned


def test_normalise_empty_returns_empty():
    assert _normalise("") == ""
    assert _normalise(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------- loader
def test_loader_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_pdf_chunks(tmp_path / "does_not_exist.pdf")


def test_loader_basic_roundtrip(tiny_pdf: Path):
    pytest.importorskip("pypdf")
    pytest.importorskip("langchain_text_splitters")
    result = load_pdf_chunks(tiny_pdf, chunk_size=200, chunk_overlap=20)
    assert isinstance(result, PDFLoaderResult)
    assert result.page_count == 2
    assert result.chunk_size == 200
    assert result.chunk_overlap == 20
    assert len(result.chunks) >= 1
    # Every chunk has a paired source pointer
    assert len(result.chunks) == len(result.sources)
    for src in result.sources:
        assert src["page"] in (1, 2)
        assert src["char_count"] > 0


def test_loader_metadata_is_compact(tiny_pdf: Path):
    pytest.importorskip("pypdf")
    pytest.importorskip("langchain_text_splitters")
    result = load_pdf_chunks(tiny_pdf, chunk_size=512, chunk_overlap=50)
    meta = result.metadata
    assert set(meta) == {"page_count", "chunks", "chunk_size", "chunk_overlap", "raw_text_chars"}
    assert meta["chunks"] == len(result.chunks)


def test_loader_filters_tiny_chunks(tiny_pdf: Path):
    pytest.importorskip("pypdf")
    pytest.importorskip("langchain_text_splitters")
    # Very high min_chunk_chars should drop everything; very low keeps all.
    big = load_pdf_chunks(tiny_pdf, chunk_size=512, min_chunk_chars=1)
    small = load_pdf_chunks(tiny_pdf, chunk_size=512, min_chunk_chars=10_000)
    assert len(big.chunks) >= len(small.chunks)
