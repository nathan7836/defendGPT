#!/usr/bin/env python3
"""Convert Instagram inbox JSON exports to plain text conversations."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "Instagram"
OUTPUT_DIR = ROOT / "data_clean" / "instagram"
SELF_NAMES = {"Anis Ayari", "Anis AYARI", "Anis AYARI (You)", "Anis"}


def _normalise_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.replace("\r", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE)
    if any(mark in text for mark in ("Ã", "â")):
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return text.strip()


def _iter_message_files(thread_dir: Path) -> Iterable[Path]:
    yield from sorted(thread_dir.glob("message_*.json"))


def _load_messages(message_file: Path) -> List[Tuple[int, str, str]]:
    payload = json.loads(message_file.read_text(encoding="utf-8"))
    result: List[Tuple[int, str, str]] = []
    for item in payload.get("messages", []):
        content = item.get("content")
        if not content:
            continue
        sender = item.get("sender_name", "").strip()
        timestamp = int(item.get("timestamp_ms", 0))
        result.append((timestamp, sender, content))
    return result


def export_instagram_inbox() -> int:
    inbox = DATA_ROOT / "inbox"
    if not inbox.exists():
        return 0
    transcripts: List[str] = []
    for thread_dir in sorted(inbox.iterdir()):
        if not thread_dir.is_dir():
            continue
        entries: List[Tuple[int, str, str]] = []
        for json_path in _iter_message_files(thread_dir):
            entries.extend(_load_messages(json_path))
        if not entries:
            continue
        entries.sort(key=lambda item: item[0])
        lines: List[str] = []
        for _, sender, content in entries:
            text = _normalise_text(content)
            if not text:
                continue
            prefix = "Assistant : " if sender in SELF_NAMES else "User : "
            lines.append(prefix + text)
        if lines:
            transcripts.append("\n".join(lines))
    if not transcripts:
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "messages.txt").write_text("\n\n".join(transcripts) + "\n", encoding="utf-8")
    return len(transcripts)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_comments(obj) -> Iterable[str]:
    if isinstance(obj, dict):
        if "string_map_data" in obj and isinstance(obj["string_map_data"], dict):
            comment_entry = obj["string_map_data"].get("Comment")
            if isinstance(comment_entry, dict):
                value = comment_entry.get("value")
                if value:
                    yield value
        for key, value in obj.items():
            if key in {"comments_story_comments", "comments_reels_comments", "comments_media_comments"}:
                yield from _extract_comments(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _extract_comments(item)


def export_instagram_comments() -> int:
    comments_dir = DATA_ROOT / "comments"
    if not comments_dir.exists():
        return 0
    lines: List[str] = []
    for path in sorted(comments_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except json.JSONDecodeError:
            continue
        for comment in _extract_comments(payload):
            norm = _normalise_text(comment)
            if norm:
                lines.append(norm)
    if not lines:
        return 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def prepare_instagram() -> Dict[str, int]:
    conversations = export_instagram_inbox()
    comments = export_instagram_comments()
    return {"conversations": conversations, "comments": comments}


def main() -> None:
    summary = prepare_instagram()
    if summary:
        lines = [f"{k}: {v}" for k, v in summary.items()]
        print("Instagram data prepared -> " + ", ".join(lines))


if __name__ == "__main__":
    main()
