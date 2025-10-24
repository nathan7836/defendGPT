#!/usr/bin/env python3
"""Streaming downloader for OSCAR corpora (language-specific)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional


@dataclass
class OscarConfig:
    dataset: str = "oscar-corpus/OSCAR-23.01"
    language: str = "fr"
    split: str = "train"
    output_dir: Path = Path("data_clean") / "oscar_fr"
    max_docs: Optional[int] = None
    max_bytes: Optional[int] = None
    min_chars: int = 50
    progress_interval: int = 1000


def _ensure_datasets_installed() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Le paquet 'datasets' est requis pour télécharger OSCAR. "
            "Installez-le via 'pip install datasets'."
        ) from exc


def _stream_oscar(cfg: OscarConfig) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(cfg.dataset, cfg.language, split=cfg.split, streaming=True)


def download_oscar(
    cfg: Optional[OscarConfig] = None,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    _ensure_datasets_installed()

    cfg = cfg or OscarConfig()
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_iter = _stream_oscar(cfg)

    slug = cfg.dataset.split("/")[-1].replace(".", "_")
    output_name = f"oscar_{slug}_{cfg.language}"
    output_path = cfg.output_dir / f"{output_name}.txt"
    metadata_path = cfg.output_dir / f"{output_name}.meta.json"

    processed = 0
    kept = 0
    bytes_written = 0
    skipped_short = 0
    start_ts = time.time()

    with output_path.open("w", encoding="utf-8") as handle:
        sentinel = "\n\n<|doc|>\n\n"
        for example in dataset_iter:
            processed += 1
            text = (
                example.get("text")
                or example.get("content")
                or example.get("document")
                or example.get("content_text")
            )
            if not text:
                continue
            text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
            if len(text) < cfg.min_chars:
                skipped_short += 1
                continue

            if kept > 0:
                handle.write(sentinel)
            handle.write(text)
            kept += 1
            bytes_written = handle.tell()

            if cfg.max_docs and kept >= cfg.max_docs:
                break
            if cfg.max_bytes and bytes_written >= cfg.max_bytes:
                break

            if kept % max(cfg.progress_interval, 1) == 0:
                payload = {
                    "processed": processed,
                    "kept_docs": kept,
                    "skipped_short": skipped_short,
                    "bytes_written": bytes_written,
                    "output_path": str(output_path),
                }
                if progress_cb:
                    progress_cb(payload)
                else:
                    print(
                        f"[oscar] kept={kept} processed={processed} bytes={bytes_written/1e6:.2f}MB",
                        flush=True,
                    )

    elapsed = time.time() - start_ts
    summary = {
        "dataset": cfg.dataset,
        "language": cfg.language,
        "split": cfg.split,
        "processed": processed,
        "kept_docs": kept,
        "skipped_short": skipped_short,
        "bytes_written": bytes_written,
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "elapsed_seconds": round(elapsed, 2),
        "max_docs": cfg.max_docs,
        "max_bytes": cfg.max_bytes,
        "min_chars": cfg.min_chars,
    }

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if progress_cb:
        progress_cb(summary)
    else:
        print(
            f"[oscar] completed kept={kept} processed={processed} bytes={bytes_written/1e6:.2f}MB",
            flush=True,
        )

    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> OscarConfig:
    parser = argparse.ArgumentParser(description="Télécharger un sous-corpus OSCAR")
    parser.add_argument("--dataset", default=OscarConfig.dataset)
    parser.add_argument("--language", default=OscarConfig.language)
    parser.add_argument("--split", default=OscarConfig.split)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OscarConfig.output_dir,
        help="Répertoire cible (défaut: data_clean/oscar_fr)",
    )
    parser.add_argument("--max-docs", type=int, default=OscarConfig.max_docs)
    parser.add_argument(
        "--max-mib",
        type=float,
        default=None,
        help="Limite approximative en MiB (optionnel)",
    )
    parser.add_argument("--min-chars", type=int, default=OscarConfig.min_chars)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=OscarConfig.progress_interval,
        help="Fréquence des logs (en documents)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    max_bytes = int(args.max_mib * 1024 * 1024) if args.max_mib else None

    return OscarConfig(
        dataset=args.dataset,
        language=args.language,
        split=args.split,
        output_dir=args.output_dir,
        max_docs=args.max_docs if (args.max_docs or 0) > 0 else None,
        max_bytes=max_bytes,
        min_chars=max(0, args.min_chars),
        progress_interval=max(1, args.progress_interval),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        cfg = parse_args(argv)
        summary = download_oscar(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

