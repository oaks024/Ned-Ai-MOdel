"""PDF text extraction.

Uses pdfplumber, which handles most NED prospectus / notice PDFs well.
Each page becomes its own document so we can cite a precise page number.
"""
import os
import pdfplumber

from src.utils import get_logger

logger = get_logger("pdf_loader")


def extract_pdf_pages(pdf_path: str, source_url: str | None = None,
                      title: str | None = None) -> list[dict]:
    """Return one record per non-empty page.

    Each record contains: page_number, text, source_url, title, file_name.
    """
    out: list[dict] = []
    if not os.path.exists(pdf_path):
        logger.warning(f"Missing PDF: {pdf_path}")
        return out

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Failed page {i} of {pdf_path}: {e}")
                    text = ""
                if not text.strip():
                    continue
                out.append({
                    "page_number": i,
                    "text": text,
                    "source_url": source_url or pdf_path,
                    "title": title or os.path.basename(pdf_path),
                    "file_name": os.path.basename(pdf_path),
                })
    except Exception as e:
        logger.warning(f"Could not open {pdf_path}: {e}")
    return out
