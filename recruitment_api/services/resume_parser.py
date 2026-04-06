import io
import pdfplumber
from docx import Document


def extract_text_from_pdf(data: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def extract_text(data: bytes, filename: str) -> str:
    """Return plain text from a PDF, DOCX, or TXT file."""
    low = filename.lower()
    if low.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if low.endswith(".docx"):
        return extract_text_from_docx(data)
    if low.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()
    return ""