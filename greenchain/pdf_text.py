
from __future__ import annotations

import io
import tempfile
from typing import Optional

def extract_text_from_pdf(path_or_bytes: str | bytes) -> str:
    """
    Use a simple, light 'pdfloader' style extraction.
    Tries langchain_community.PyPDFLoader if available; otherwise falls back to pypdf.
    """
    text = None

    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
        if isinstance(path_or_bytes, (str,)):
            loader = PyPDFLoader(path_or_bytes)
        else:
            # write bytes to temp file for loader
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(path_or_bytes)
                tmp.flush()
                loader = PyPDFLoader(tmp.name)
        docs = loader.load()
        text = "\n\n".join([d.page_content for d in docs])
        if text:
            return text
    except Exception:
        pass

    try:
        from pypdf import PdfReader  # lightweight fallback
        if isinstance(path_or_bytes, (str,)):
            reader = PdfReader(path_or_bytes)
        else:
            reader = PdfReader(io.BytesIO(path_or_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")
