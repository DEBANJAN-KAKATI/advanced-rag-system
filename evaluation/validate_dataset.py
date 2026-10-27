"""Checks that every evidence span in the golden set exists verbatim in the corpus.

Usage:
    python -m evaluation.validate_dataset [--dataset path]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from evaluation.dataset import DEFAULT_DATASET, load_corpus, load_dataset, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    problems = validate(items, load_corpus())
    counts = Counter(i.type for i in items)
    print(f"{len(items)} questions: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for p in problems:
        print("  PROBLEM", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
