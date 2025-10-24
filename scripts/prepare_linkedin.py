#!/usr/bin/env python3
"""Normalise Linkedin CSV exports into plain-text corpora."""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "Linkedin-data"
OUTPUT_DIR = ROOT / "data_clean" / "linkedin"
SELF_NAME = "Anis Ayari"


def _normalise_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace('"', '')
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
    return text.strip()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def export_comments() -> int:
    src = DATA_ROOT / "Comments.csv"
    if not src.exists():
        return 0
    rows = _read_csv(src)
    lines: List[str] = []
    for row in rows:
        message = _normalise_text(row.get("Message"))
        if message:
            lines.append(message)
    if not lines:
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def export_shares() -> int:
    src = DATA_ROOT / "Shares.csv"
    if not src.exists():
        return 0
    rows = _read_csv(src)
    lines: List[str] = []
    for row in rows:
        commentary = _normalise_text(row.get("ShareCommentary"))
        if commentary:
            lines.append(commentary)
    if not lines:
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "shares.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _parse_date(value: str | None) -> datetime:
    if not value:
        return datetime.min
    value = value.strip()
    # LinkedIn export often uses "YYYY-MM-DD HH:MM:SS UTC"
    for fmt in ("%Y-%m-%d %H:%M:%S %Z", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.min


def export_messages() -> int:
    src = DATA_ROOT / "messages.csv"
    if not src.exists():
        return 0
    rows = _read_csv(src)
    conversations: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        conv_id = row.get("CONVERSATION ID") or "unknown"
        conversations[conv_id].append(row)

    transcripts: List[str] = []
    for conv_id, messages in conversations.items():
        messages.sort(key=lambda item: _parse_date(item.get("DATE")))
        lines: List[str] = []
        for message in messages:
            speaker = message.get("FROM", "").strip() or ""
            content = _normalise_text(message.get("CONTENT") or message.get("SUBJECT"))
            if not content:
                continue
            if speaker == SELF_NAME:
                prefix = "Assistant : "
            else:
                prefix = "User : "
            lines.append(f"{prefix}{content}")
        if lines:
            transcripts.append("\n".join(lines))

    if not transcripts:
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "messages.txt").write_text("\n\n".join(transcripts) + "\n", encoding="utf-8")
    return len(transcripts)


def prepare_linkedin() -> Dict[str, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comments = export_comments()
    shares = export_shares()
    messages = export_messages()
    return {
        "comments": comments,
        "shares": shares,
        "conversations": messages,
    }


def main() -> None:
    summary = prepare_linkedin()
    if summary:
        lines = [f"{key}: {value}" for key, value in summary.items()]
        print("LinkedIn data prepared -> " + ", ".join(lines))


if __name__ == "__main__":
    main()
