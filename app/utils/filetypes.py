import os

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".md": "txt",
    ".py": "txt",
    ".json": "txt",
    ".csv": "txt"
}

def get_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return SUPPORTED_EXTENSIONS.get(ext, "txt")

def is_supported_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS
