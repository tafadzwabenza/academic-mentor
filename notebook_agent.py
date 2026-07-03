from io import BytesIO
import zipfile
from typing import Any, Dict, Union
from xml.etree import ElementTree

from PyPDF2 import PdfReader

import database_manager as dbm

UploadedFile = Union[Any, BytesIO]

TEXT_EXTENSIONS = {".txt", ".csv"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".mp4", ".mp3", ".wav"}


def _file_bytes(uploaded_file: UploadedFile) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    if hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return data
    raise ValueError("uploaded_file must be a Streamlit UploadedFile or file-like object")


def _file_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def extract_text_from_pdf(uploaded_file: UploadedFile) -> str:
    filename = getattr(uploaded_file, "name", "upload.pdf")
    pdf_bytes = _file_bytes(uploaded_file)

    if not pdf_bytes:
        raise ValueError(f"No PDF data found in {filename}")

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(f"Could not extract text from {filename}")

    return text


def extract_text_from_docx(uploaded_file: UploadedFile) -> str:
    filename = getattr(uploaded_file, "name", "upload.docx")
    docx_bytes = _file_bytes(uploaded_file)

    with zipfile.ZipFile(BytesIO(docx_bytes)) as archive:
        xml_content = archive.read("word/document.xml")

    root = ElementTree.fromstring(xml_content)
    parts = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            parts.append(node.text)

    text = " ".join(parts).strip()
    if not text:
        raise ValueError(f"Could not extract text from {filename}")

    return text


def extract_text_from_plaintext(uploaded_file: UploadedFile) -> str:
    filename = getattr(uploaded_file, "name", "upload.txt")
    raw = _file_bytes(uploaded_file)
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"No readable text found in {filename}")
    return text


def extract_text_from_file(uploaded_file: UploadedFile) -> str:
    filename = getattr(uploaded_file, "name", "upload")
    extension = _file_extension(filename)

    if extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    if extension == "docx":
        return extract_text_from_docx(uploaded_file)
    if extension in {"txt", "csv"}:
        return extract_text_from_plaintext(uploaded_file)
    if extension in {"png", "jpg", "jpeg", "mp4", "mp3", "wav"}:
        size_kb = round(len(_file_bytes(uploaded_file)) / 1024, 1)
        return (
            f"[Media attachment: {filename} ({extension.upper()}, {size_kb} KB). "
            "Stored for reference — ask me about this file in chat.]"
        )

    raise ValueError(f"Unsupported file type: {filename}")


def _build_preview_summary(text: str, max_length: int = 500) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[: max_length - 3].rstrip() + "..."


def process_pdf_upload(uploaded_file: UploadedFile, project_id: int) -> Dict[str, Any]:
    return process_file_upload(uploaded_file, project_id)


def process_file_upload(uploaded_file: UploadedFile, project_id: int) -> Dict[str, Any]:
    filename = getattr(uploaded_file, "name", "upload")
    extracted_text = extract_text_from_file(uploaded_file)
    preview_summary = _build_preview_summary(extracted_text)
    source_id = dbm.save_source(
        project_id=project_id,
        filename=filename,
        extracted_text=extracted_text,
        preview_summary=preview_summary,
    )
    return {
        "source_id": source_id,
        "filename": filename,
        "preview_summary": preview_summary,
        "char_count": len(extracted_text),
    }
