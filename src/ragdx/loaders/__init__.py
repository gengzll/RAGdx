"""Document loaders that produce ``corpus_chunks`` for the RAG pipeline."""

from ragdx.loaders.pdf import PDFLoaderResult, load_pdf_chunks

__all__ = ["PDFLoaderResult", "load_pdf_chunks"]
