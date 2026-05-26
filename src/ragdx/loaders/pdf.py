"""PDF loader: PDF file -> ``list[chunk_str]`` ready for the RAG pipeline.

Uses ``pypdf`` for page extraction (no system deps, ships with langchain
runtime) and LangChain's ``RecursiveCharacterTextSplitter`` for chunking.
Chunk size + overlap are tunable so they can drive an AutoRAG search.

The result also carries the per-chunk source pointer (page + chunk index)
which is useful when you want to attribute an answer back to a page
range in the original document.

Example::

    from ragdx.loaders import load_pdf_chunks

    res = load_pdf_chunks("report.pdf", chunk_size=512, chunk_overlap=50)
    print(f"{res.page_count} pages -> {len(res.chunks)} chunks")
    rag_corpus = res.chunks  # list[str]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PDFLoaderResult:
    """Output of :func:`load_pdf_chunks`. Drop-in for the RAG pipeline."""

    chunks: list[str]
    sources: list[dict]  # one per chunk: {page, chunk_index, char_start, char_end}
    page_count: int
    chunk_size: int
    chunk_overlap: int
    raw_text_chars: int

    def __post_init__(self) -> None:  # pragma: no cover - sanity check
        assert len(self.chunks) == len(self.sources)

    @property
    def metadata(self) -> dict:
        """Compact summary suitable for logging or saving with the bundle."""
        return {
            "page_count": self.page_count,
            "chunks": len(self.chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "raw_text_chars": self.raw_text_chars,
        }


_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse runs of whitespace and strip control chars that PDFs leak."""
    if not text:
        return ""
    cleaned = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 32)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _read_pdf_pages(path: Path) -> list[str]:
    """Extract per-page text using pypdf. Returns raw page strings.

    pypdf's ``extract_text`` is a best-effort process for layout-heavy PDFs;
    we normalise whitespace afterwards but don't attempt OCR. For scanned
    PDFs the caller should run OCR first and feed the resulting text.
    """
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "PDF loading requires `pypdf`. Install with `pip install pypdf` "
            "(it's already pulled in by the langchain extras)."
        ) from exc

    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def load_pdf_chunks(
    path: str | Path,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 50,
) -> PDFLoaderResult:
    """Read a PDF and split it into chunks suitable for vector indexing.

    Parameters
    ----------
    path:
        Filesystem path to the PDF.
    chunk_size / chunk_overlap:
        Forwarded to LangChain's ``RecursiveCharacterTextSplitter``. Both
        are measured in characters (a rough proxy for tokens that doesn't
        need a tokenizer). Tunable so they can be part of an AutoRAG search.
    min_chunk_chars:
        Discard tiny chunks (e.g. page headers / blank pages) below this
        length. Helps avoid 1-line noise dominating the retriever.

    Returns
    -------
    PDFLoaderResult — ``.chunks`` is the list[str] you feed into FAISS,
    ``.sources`` carries page + chunk index per chunk so you can map
    answers back to the original document.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception as exc:  # pragma: no cover - depends on user env
        raise ImportError(
            "PDF chunking requires langchain_text_splitters. Install with "
            "`pip install langchain-text-splitters` (also bundled with the "
            "ragdx[langchain] extra)."
        ) from exc

    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = _read_pdf_pages(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    chunks: list[str] = []
    sources: list[dict] = []
    raw_chars = 0
    for page_idx, raw_text in enumerate(pages):
        normalised = _normalise(raw_text)
        raw_chars += len(raw_text)
        if not normalised:
            continue
        for i, c in enumerate(splitter.split_text(normalised)):
            if len(c) < min_chunk_chars:
                continue
            chunks.append(c)
            sources.append(
                {
                    "page": page_idx + 1,
                    "chunk_index": i,
                    "char_count": len(c),
                }
            )

    return PDFLoaderResult(
        chunks=chunks,
        sources=sources,
        page_count=len(pages),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        raw_text_chars=raw_chars,
    )


__all__ = ["PDFLoaderResult", "load_pdf_chunks"]
