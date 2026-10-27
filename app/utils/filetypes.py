import os

# Extension -> parser key. Legacy binary `.doc` files are intentionally not listed:
# python-docx can only read the modern `.docx` format.
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".md": "txt",
    ".markdown": "txt",
    ".rst": "txt",
    ".log": "txt",
    ".py": "txt",
    ".json": "json",
    ".csv": "csv",
}


def get_extension(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def get_file_type(filename: str) -> str:
    return SUPPORTED_EXTENSIONS.get(get_extension(filename), "txt")


def is_supported_file(filename: str) -> bool:
    return get_extension(filename) in SUPPORTED_EXTENSIONS


def supported_extensions_label() -> str:
    return ", ".join(sorted(ext.lstrip(".").upper() for ext in SUPPORTED_EXTENSIONS))
