#!/usr/bin/env python3
"""Convert YouTube .vtt caption files to cleaned plain text."""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data"
DATA_CLEAN_ROOT = ROOT / "data_clean"
RAW_SUBTITLE_DIR = DATA_ROOT / "youtube-subtitle"
PLAIN_SUBTITLE_DIR = DATA_ROOT / "youtube-subtitile-plain"
CHANNELS_SOURCE_ROOT = DATA_ROOT / "youtube_channels"
CLEAN_DATA_DIR = DATA_CLEAN_ROOT / "youtube"
HISTORY_DIR = DATA_CLEAN_ROOT / "_history"

# Backwards compatibility for dashboard imports
SOURCE_DIR = RAW_SUBTITLE_DIR
TARGET_DIR = PLAIN_SUBTITLE_DIR

TAG_PATTERN = re.compile(r"<[^>]+>")
TIMESTAMP_PATTERN = re.compile(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->")
SKIP_PREFIXES = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")
NOISE_PREFIXES = ("align:", "position:", "line:", "size:", "vertical:")
STAGE_DIRECTION_PATTERN = re.compile(r"^\[[^\]]+\]$")


def make_progress(total: int, *, desc: str, unit: str = "fichier", leave: bool = False) -> Optional[tqdm]:
    if total <= 0 or not sys.stderr.isatty():
        return None
    return tqdm(total=total, desc=desc, unit=unit, leave=leave)


def clean_line(raw: str) -> str | None:
    """Strip timestamps and markup, returning spoken text or None."""
    stripped = raw.strip()
    if not stripped:
        return None
    if any(stripped.startswith(prefix) for prefix in SKIP_PREFIXES):
        return None
    if TIMESTAMP_PATTERN.search(stripped):
        return None
    if any(stripped.startswith(prefix) for prefix in NOISE_PREFIXES):
        return None

    text = TAG_PATTERN.sub("", raw)
    text = text.replace("\u200b", "")
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return None
    if STAGE_DIRECTION_PATTERN.match(text):
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def convert_file(
    vtt_path: Path,
    target_dir: Path = PLAIN_SUBTITLE_DIR,
    *,
    name_prefix: str = "",
) -> Path:
    lines: list[str] = []
    previous: str | None = None

    for raw in vtt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        cleaned = clean_line(raw)
        if cleaned is None:
            continue
        if cleaned == previous:
            continue
        lines.append(cleaned)
        previous = cleaned

    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{name_prefix}{vtt_path.stem}.txt"
    target_path = target_dir / filename
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_path


def convert_directory(
    source_dir: Path,
    target_dir: Path = PLAIN_SUBTITLE_DIR,
    *,
    name_prefix: str = "",
) -> dict[str, object]:
    """Convert every .vtt file within source_dir into cleaned .txt in target_dir."""

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if not source_dir.exists():
        return {
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "converted_files": 0,
            "total_vtt": 0,
        }
    vtt_files = sorted(source_dir.glob("*.vtt"))
    bar = make_progress(len(vtt_files), desc=f"Nettoyage {source_dir.name}")
    converted = 0
    for vtt_file in vtt_files:
        convert_file(vtt_file, target_dir=target_dir, name_prefix=name_prefix)
        converted += 1
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "converted_files": converted,
        "total_vtt": len(vtt_files),
    }


def _iter_channel_subtitle_dirs(root: Path = CHANNELS_SOURCE_ROOT) -> list[tuple[str, Path]]:
    if not root.exists():
        return []
    entries: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for subtitles_dir in root.rglob("subtitles"):
        if not subtitles_dir.is_dir():
            continue
        resolved = subtitles_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        slug = subtitles_dir.parent.name
        if not slug:
            continue
        entries.append((slug, subtitles_dir))
    entries.sort(key=lambda item: (item[0], str(item[1])))
    return entries


def sync_channel_vtts(
    fn_root: Path = CHANNELS_SOURCE_ROOT,
    destination: Path = RAW_SUBTITLE_DIR,
) -> dict[str, object]:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    moved = 0
    skipped = 0
    per_channel: dict[str, int] = {}

    tasks: list[tuple[str, Path]] = []
    for slug, subtitles_dir in _iter_channel_subtitle_dirs(fn_root):
        for vtt_path in sorted(subtitles_dir.glob("*.vtt")):
            tasks.append((slug, vtt_path))

    bar = make_progress(len(tasks), desc="Synchronisation VTT", leave=False)

    for slug, vtt_path in tasks:
        target_name = f"{slug}__{vtt_path.name}"
        target_path = destination / target_name
        if target_path.exists():
            skipped += 1
        else:
            try:
                shutil.move(str(vtt_path), str(target_path))
            except OSError:
                skipped += 1
            else:
                moved += 1
                per_channel[slug] = per_channel.get(slug, 0) + 1
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    return {
        "destination": str(destination),
        "moved_files": moved,
        "skipped_files": skipped,
        "per_channel": per_channel,
    }


def convert_all_sources() -> dict[str, object]:
    summaries: dict[str, object] = {}
    total_converted = 0

    sync_summary = sync_channel_vtts()
    summaries["sync"] = sync_summary

    convert_summary = convert_directory(RAW_SUBTITLE_DIR, PLAIN_SUBTITLE_DIR)
    summaries["convert"] = convert_summary
    total_converted += convert_summary["converted_files"]

    mirror_summary = mirror_plain_to_clean()
    summaries["mirror"] = mirror_summary

    summaries["total_converted"] = total_converted
    return summaries


def mirror_plain_to_clean(
    source_dir: Path = PLAIN_SUBTITLE_DIR, destination: Path = CLEAN_DATA_DIR
) -> dict[str, object]:
    source_dir = Path(source_dir)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        return {"source_dir": str(source_dir), "destination": str(destination), "copied": 0}

    txt_files = sorted(source_dir.glob("*.txt"))
    bar = make_progress(len(txt_files), desc="Copie vers data_clean", leave=False)
    copied = 0
    for txt_path in txt_files:
        target_path = destination / txt_path.name
        target_path.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1
        if bar is not None:
            bar.update(1)
    if bar is not None:
        bar.close()
    return {"source_dir": str(source_dir), "destination": str(destination), "copied": copied}


def main() -> None:
    started = datetime.utcnow()
    summaries = convert_all_sources()
    converted = summaries.get("total_converted", 0)
    print(f"Nettoyage terminé: {converted} fichiers convertis.")
    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": "subtitles",
            "status": "completed",
            "started_at": started.isoformat(),
            "ended_at": datetime.utcnow().isoformat(),
            "details": summaries,
        }
        history_path = HISTORY_DIR / "subtitles.json"
        history_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


if __name__ == "__main__":
    main()
