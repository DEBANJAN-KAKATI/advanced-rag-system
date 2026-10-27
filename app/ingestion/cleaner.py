import re

# Lines that are only page furniture: "Page 3", "Page 3 of 10", "3 of 10", "- 3 -", "--- Page 3 ---".
# Bare numbers are deliberately kept: they are often table values, not page numbers.
_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:[-=_]{2,}\s*)?(?:page\s+\d+(?:\s+of\s+\d+)?|\d+\s+of\s+\d+|-\s*\d+\s*-)(?:\s*[-=_]{2,})?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_text(text: str) -> str:
    """Clean and normalize extracted document text while preserving paragraph structure."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("­", "")  # non-breaking space, soft hyphen
    text = re.sub(r"[​-‍⁠﻿]", "", text)  # zero-width characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)  # control characters

    text = _PAGE_MARKER_RE.sub("", text)

    # Collapse dot leaders from tables of contents ("Introduction ........ 4")
    text = re.sub(r"(?:\.\s*){5,}", "... ", text)

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def fix_pdf_line_breaks(text: str) -> str:
    """
    Undoes PDF line wrapping: rejoins words hyphenated across lines ("exam-\\nple") and
    joins soft-wrapped lines (no ending punctuation, next line starts lowercase) into one line.
    """
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    return re.sub(r"(?<=[A-Za-z0-9,'\")])[ \t]*\n(?=[a-z(])", " ", text)
