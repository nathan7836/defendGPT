#!/usr/bin/env python3
"""Utilities to clean previously downloaded Wikipedia articles."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data_clean" / "wikipedia"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

# Regex patterns that capture most LaTeX/math artefacts present in Wikipedia extracts.
LATEX_BLOCK_PATTERNS = [
    re.compile(pattern, flags=re.DOTALL)
    for pattern in (
        r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}",
        r"\\\[.*?\\\]",
        r"\\\(.*?\\\)",
        r"\$\$.*?\$\$",
        r"\$[^$]*?\$",
        r"\{\\?displaystyle[^}]*\}",
    )
]

# Simple patterns that are easier to replace globally without risking valid prose.
LATEX_INLINE_REPLACEMENTS = [
    (re.compile(r"\\mathrm\{([^}]+)\}"), r"\1"),
    (re.compile(r"\\operatorname\{([^}]+)\}"), r"\1"),
    (re.compile(r"\\text\{([^}]+)\}"), r"\1"),
]

LATEX_COMMAND_PATTERN = re.compile(r"\\[a-zA-Z]+\*?", flags=re.UNICODE)
BRACES_PATTERN = re.compile(r"[{}]")
REFERENCE_PATTERN = re.compile(r"\[[0-9]+\]")
WHITESPACE_PATTERN = re.compile(r"\s+")
EMPTY_PARENS_PATTERN = re.compile(r"\(\s*\)")


@dataclass
class CleanStats:
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    original_bytes: int = 0
    cleaned_bytes: int = 0
    removed_lines: int = 0
    kept_lines: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "skipped_files": self.skipped_files,
            "original_bytes": self.original_bytes,
            "cleaned_bytes": self.cleaned_bytes,
            "removed_lines": self.removed_lines,
            "kept_lines": self.kept_lines,
        }


def _strip_latex_blocks(text: str) -> str:
    for pattern in LATEX_BLOCK_PATTERNS:
        text = pattern.sub(" ", text)
    for pattern, replacement in LATEX_INLINE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    text = LATEX_COMMAND_PATTERN.sub(" ", text)
    text = BRACES_PATTERN.sub(" ", text)
    text = REFERENCE_PATTERN.sub(" ", text)
    text = EMPTY_PARENS_PATTERN.sub(" ", text)
    return text


def clean_article_text(raw_text: str) -> tuple[str, int]:
    """Return a cleaned version of an article and number of removed lines."""

    cleaned_lines = []
    removed = 0
    for line in raw_text.splitlines():
        line = _strip_latex_blocks(line)
        line = WHITESPACE_PATTERN.sub(" ", line).strip()
        if not line:
            removed += 1
            continue
        # Drop lines that still look like formulas or templates after stripping.
        if _looks_like_formula(line):
            removed += 1
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return "", removed
    return "\n".join(cleaned_lines).strip() + "\n", removed


def _looks_like_formula(line: str) -> bool:
    # Heuristics: too many non-letter symbols or residual underscores/hat characters.
    if not line:
        return True
    letters = sum(1 for char in line if char.isalpha())
    dense_symbols = sum(1 for char in line if char in "_`^=<>|\\")
    if letters == 0:
        return True
    if dense_symbols > letters:
        return True
    if "\frac" in line or "\sum" in line or "\int" in line:
        return True
    return False


def iter_text_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.txt")):
        if path.is_file():
            yield path


def clean_wikipedia_corpus(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Optional[Path] = None,
    *,
    max_files: Optional[int] = None,
    progress_cb: Optional[Callable[[Dict[str, int | str]], None]] = None,
) -> Dict[str, int | str]:
    """Clean all .txt articles in ``input_dir`` and write to ``output_dir``."""

    output_dir = output_dir or input_dir
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_text_files(input_dir))
    if max_files is not None:
        files = files[:max_files]

    stats = CleanStats(total_files=len(files))
    start_time = time.time()

    for index, path in enumerate(files, 1):
        raw_text = path.read_text(encoding="utf-8")
        original_bytes = len(raw_text.encode("utf-8"))
        cleaned_text, removed_lines = clean_article_text(raw_text)
        stats.removed_lines += removed_lines
        if not cleaned_text:
            stats.skipped_files += 1
            if progress_cb:
                progress_cb({
                    "processed": index,
                    "total": stats.total_files,
                    "last_file": path.name,
                    "skipped": stats.skipped_files,
                    "total_files": stats.total_files,
                })
            continue

        output_path = output_dir / path.name
        tmp_path = output_path.with_suffix(".tmp")
        tmp_path.write_text(cleaned_text, encoding="utf-8")
        tmp_path.replace(output_path)

        stats.processed_files += 1
        stats.original_bytes += original_bytes
        stats.cleaned_bytes += len(cleaned_text.encode("utf-8"))
        stats.kept_lines += cleaned_text.count("\n") + 1

        if progress_cb:
            progress_cb(
                {
                    "processed": index,
                    "total": stats.total_files,
                    "last_file": path.name,
                    "last_output": str(output_path),
                    "skipped": stats.skipped_files,
                    "total_files": stats.total_files,
                }
            )

    duration = int(time.time() - start_time)
    summary: Dict[str, int | str] = stats.to_dict()
    summary["elapsed_seconds"] = duration
    summary["input_dir"] = str(input_dir)
    summary["output_dir"] = str(output_dir)
    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean downloaded Wikipedia articles")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Limit processing to the first N files")
    parser.add_argument("--stats", type=Path, default=None, help="Optional path to save cleaning summary as JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    summary = clean_wikipedia_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
    )
    if args.stats:
        args.stats.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
