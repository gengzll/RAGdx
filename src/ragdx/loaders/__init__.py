"""Document loaders that produce ``corpus_chunks`` for the RAG pipeline,
plus ground-truth table loaders for the with-GT experiment path."""

from ragdx.loaders.pdf import PDFLoaderResult, load_pdf_chunks
from ragdx.loaders.tabular import (
    detect_mapping,
    load_gt_table,
    missing_required,
    records_from_table,
    write_questions_jsonl,
)

__all__ = [
    "PDFLoaderResult",
    "detect_mapping",
    "load_gt_table",
    "load_pdf_chunks",
    "missing_required",
    "records_from_table",
    "write_questions_jsonl",
]
