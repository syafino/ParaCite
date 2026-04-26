"""Plain-text extraction for uploaded files.

Dispatches on file suffix:
- ``.txt`` / ``.md`` -> ``Path.read_text``
- ``.pdf``           -> ``pypdf.PdfReader`` page loop
- anything else      -> ``UnsupportedFileType``
"""

from __future__ import annotations

from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


class UnsupportedFileType(ValueError):
    """Raised when the uploaded file's suffix isn't in SUPPORTED_SUFFIXES."""


def extract_text(path: Path) -> str:
    """Return the plain-text content of ``path``.

    Raises ``FileNotFoundError`` if the path doesn't exist and
    ``UnsupportedFileType`` if the suffix isn't supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf(path)
    raise UnsupportedFileType(
        f"unsupported file type: {suffix!r}. "
        f"Supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - skip unparseable pages, keep going
            pages.append("")
    return "\n\n".join(p.strip() for p in pages if p.strip())
