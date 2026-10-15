"""PDF text and page metadata; textless pages are not negative evidence."""

import re
import warnings
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pypdf import PdfReader


class DocumentError(ValueError):
    pass


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Document:
    source: str
    digest: str
    pages: tuple[Page, ...]


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    text: str
    source: str
    ordinal: int


def load_pdf(path: Path) -> Document:
    if not path.is_file():
        raise DocumentError(f"PDF не найден: {path}")
    try:
        content = path.read_bytes()
        if not content:
            raise DocumentError("PDF пустой: файл содержит 0 байт")
        if len(content) > 50_000_000:
            raise DocumentError("PDF больше 50 МБ: для этой версии нужен меньший файл")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = PdfReader(path, strict=True)
            if reader.is_encrypted:
                raise DocumentError("PDF зашифрован: сохраните копию без пароля")
            if not reader.pages:
                raise DocumentError("В PDF нет страниц")
            pages = tuple(
                Page(i, re.sub(r"\s+", " ", p.extract_text() or "").strip())
                for i, p in enumerate(reader.pages, 1)
            )
        empty = [str(p.number) for p in pages if not p.text]
        if empty:
            raise DocumentError(
                "На страницах "
                + ", ".join(empty)
                + " нет извлекаемого текста. Возможен скан; нужен OCR. "
                "Это не означает, что ответа в документе нет."
            )
        return Document(path.name, sha256(content).hexdigest(), pages)
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("PDF повреждён или его текст невозможно прочитать") from None


def make_chunks(document: Document, tokenizer, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    for page in document.pages:
        # Offsets preserve exact text; decode(token IDs) may change punctuation/spacing.
        offsets = tokenizer(page.text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        start = 0
        ordinal = 0
        while start < len(offsets):
            end = min(start + size, len(offsets))
            text = page.text[offsets[start][0] : offsets[end - 1][1]].strip()
            if text:
                cid = sha256(
                    f"{document.digest}:{page.number}:{ordinal}:{text}".encode()
                ).hexdigest()[:12]
                chunks.append(
                    Chunk(f"p{page.number}-{cid}", page.number, text, document.source, ordinal)
                )
            if end == len(offsets):
                break
            start = end - overlap
            ordinal += 1
    if not chunks:
        raise DocumentError("Не удалось разделить извлечённый текст на фрагменты")
    return chunks
