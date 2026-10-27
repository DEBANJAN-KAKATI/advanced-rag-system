"""
Rebuilds the evaluation corpus from upstream Python Enhancement Proposals (PEPs).

PEPs are a good RAG test corpus: they are long, sectioned technical documents,
several of them overlap in vocabulary (PEP 8 vs PEP 257 on docstrings, PEP 484
vs PEP 526 on annotations), and every PEP used here is placed in the public
domain by its authors, so the processed text can live in this repository.

The committed files in evaluation/corpus/ are the output of this script. Only
re-run it if you want to refresh the corpus -- the golden QA set's evidence
spans are verified against the committed text (see `python -m evaluation.validate_dataset`).

Usage:
    python -m evaluation.build_corpus
"""
import re
import sys
import urllib.request
from pathlib import Path

PEPS = {
    "0008": "pep-0008-style-guide.txt",
    "0257": "pep-0257-docstring-conventions.txt",
    "0405": "pep-0405-virtual-environments.txt",
    "0484": "pep-0484-type-hints.txt",
    "0517": "pep-0517-build-system-interface.txt",
    "0526": "pep-0526-variable-annotations.txt",
    "0572": "pep-0572-assignment-expressions.txt",
    "3333": "pep-3333-wsgi.txt",
}
SOURCE_URL = "https://raw.githubusercontent.com/python/peps/main/peps/pep-{num}.rst"
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

# RST section adornment characters, in the order PEPs typically use them.
_LEVELS = {"=": 1, "-": 2, "*": 3, "~": 3, "'": 3, "^": 3, '"': 3}
_ADORNMENT = re.compile(r"([=\-~'^\"*])\1{2,}\s*")


def rst_to_text(rst: str) -> str:
    """Converts RST section titles into Markdown-style '#' headings.

    The app's structure-aware chunker detects sections by a leading '#', so
    this keeps section metadata intact without otherwise altering the prose.
    """
    lines = rst.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        after = lines[i + 2] if i + 2 < len(lines) else ""
        # Overlined title: "=====" / " Title " / "====="
        if _ADORNMENT.fullmatch(line) and nxt.strip() and _ADORNMENT.fullmatch(after):
            out.append("# " + nxt.strip())
            i += 3
            continue
        is_title = (
            line.strip()
            and not line.startswith(" ")
            and _ADORNMENT.fullmatch(nxt or "")
            and len(nxt.strip()) >= len(line.strip()) - 2
        )
        if is_title:
            level = _LEVELS.get(nxt.strip()[0], 3)
            out.append("#" * level + " " + line.strip())
            i += 2
            continue
        # Drop RST-only directive lines that carry no prose.
        if re.match(r"^\.\.\s+(_[\w -]+:|\w[\w-]*::)\s*\S*$", line.strip()) and "code" not in line:
            i += 1
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for num, filename in PEPS.items():
        url = SOURCE_URL.format(num=num)
        with urllib.request.urlopen(url, timeout=30) as resp:
            rst = resp.read().decode("utf-8")
        if "public domain" not in rst.lower():
            print(f"Refusing PEP {num}: not marked as public domain", file=sys.stderr)
            return 1
        (CORPUS_DIR / filename).write_text(rst_to_text(rst), encoding="utf-8")
        print(f"Wrote {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
