#!/usr/bin/env python3
"""Utilities to fetch a French slice of the 🍷 FineWeb dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

# Optional dependencies – imported lazily so the module can be referenced without
# pulling heavy packages during server startup.


@dataclass
class FineWebConfig:
    dataset_path: str = "HuggingFaceFW/fineweb"
    config_name: str = "sample-10BT"
    split: str = "train"
    output_dir: Path = Path("data_clean") / "fineweb_fr"
    max_docs: int = 2000
    max_bytes: Optional[int] = 200 * 1024 * 1024  # 200 MiB by default
    lang_threshold: float = 0.7
    min_chars: int = 280
    progress_interval: int = 100
    lang_detect_max_chars: int = 4000


def _ensure_dependencies() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Le paquet 'datasets' est requis pour télécharger FineWeb. "
            "Installez-le via 'pip install datasets' avant de relancer."
        ) from exc
    try:
        import langdetect  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Le paquet 'langdetect' est requis pour filtrer le français. "
            "Installez-le via 'pip install langdetect'."
        ) from exc


def _normalise_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


_LANGDETECT_INITIALISED = False


def _is_french(text: str, threshold: float, *, max_chars: Optional[int] = None) -> bool:
    from langdetect import DetectorFactory, detect_langs

    global _LANGDETECT_INITIALISED
    if not _LANGDETECT_INITIALISED:
        DetectorFactory.seed = 13  # deterministic predictions
        _LANGDETECT_INITIALISED = True

    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]

    try:
        predictions = detect_langs(text)
    except Exception:
        return False
    for prediction in predictions:
        if prediction.lang == "fr" and prediction.prob >= threshold:
            return True
    return False


def _stream_fineweb(config: FineWebConfig) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    return load_dataset(
        config.dataset_path,
        name=config.config_name,
        split=config.split,
        streaming=True,
    )


def download_fineweb_fr(
    config: Optional[FineWebConfig] = None,
    *,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Download a French-filtered slice of FineWeb.

    Parameters
    ----------
    config:
        Optional configuration overriding the defaults.
    progress_cb:
        Callback invoked periodically with basic counters.

    Returns
    -------
    dict
        Summary with processed/kept stats and paths.
    """

    _ensure_dependencies()

    cfg = config or FineWebConfig()
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_iter = _stream_fineweb(cfg)

    output_name = f"fineweb_fr_{cfg.config_name.replace('/', '_')}"
    output_path = cfg.output_dir / f"{output_name}.txt"
    metadata_path = cfg.output_dir / f"{output_name}.meta.json"

    processed = 0
    kept = 0
    skipped_short = 0
    skipped_lang = 0
    bytes_written = 0
    start_ts = time.time()

    # Ensure we don't append to a previous run by truncating the file.
    with output_path.open("w", encoding="utf-8") as handle:
        sentinel = "\n\n<|doc|>\n\n"
        for example in dataset_iter:
            if cfg.max_docs and kept >= cfg.max_docs:
                break

            text = example.get("text") or example.get("content") or example.get("document")
            if not text:
                continue
            processed += 1
            text = _normalise_text(text)
            if len(text) < cfg.min_chars:
                skipped_short += 1
                continue
            if not _is_french(text, cfg.lang_threshold, max_chars=cfg.lang_detect_max_chars):
                skipped_lang += 1
                continue

            if kept > 0:
                handle.write(sentinel)
            handle.write(text)
            kept += 1
            bytes_written = handle.tell()

            if cfg.max_bytes and bytes_written >= cfg.max_bytes:
                break

            if kept % max(1, cfg.progress_interval) == 0:
                payload = {
                    "processed": processed,
                    "kept_docs": kept,
                    "skipped_short": skipped_short,
                    "skipped_lang": skipped_lang,
                    "bytes_written": bytes_written,
                    "output_path": str(output_path),
                }
                if progress_cb:
                    progress_cb(payload)
                else:
                    print(
                        f"[fineweb] kept={kept} processed={processed} bytes={bytes_written/1e6:.2f}MB",
                        flush=True,
                    )

    elapsed = time.time() - start_ts

    summary = {
        "processed": processed,
        "kept_docs": kept,
        "skipped_short": skipped_short,
        "skipped_lang": skipped_lang,
        "bytes_written": bytes_written,
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "elapsed_seconds": round(elapsed, 2),
        "dataset_path": cfg.dataset_path,
        "config_name": cfg.config_name,
        "split": cfg.split,
        "max_docs": cfg.max_docs,
        "max_bytes": cfg.max_bytes,
    }

    with metadata_path.open("w", encoding="utf-8") as meta_handle:
        json.dump(summary, meta_handle, indent=2, ensure_ascii=False)

    if progress_cb:
        progress_cb(summary)
    else:
        print(
            f"[fineweb] completed kept={kept} processed={processed} bytes={bytes_written/1e6:.2f}MB",
            flush=True,
        )

    return summary


def parse_args(argv: Optional[Iterable[str]] = None) -> FineWebConfig:
    parser = argparse.ArgumentParser(description="Télécharger un extrait FR de FineWeb")
    parser.add_argument("--dataset", dest="dataset_path", default=FineWebConfig.dataset_path)
    parser.add_argument("--config", dest="config_name", default=FineWebConfig.config_name)
    parser.add_argument("--split", default=FineWebConfig.split)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FineWebConfig.output_dir,
        help="Répertoire de sortie (défaut: data_clean/fineweb_fr)",
    )
    parser.add_argument("--max-docs", type=int, default=FineWebConfig.max_docs)
    parser.add_argument(
        "--max-mib",
        type=float,
        default=FineWebConfig.max_bytes / (1024 * 1024)
        if FineWebConfig.max_bytes
        else None,
        help="Limite approximative en MiB",
    )
    parser.add_argument(
        "--lang-threshold",
        type=float,
        default=FineWebConfig.lang_threshold,
        help="Probabilité minimale (langdetect) pour garder un document FR.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=FineWebConfig.min_chars,
        help="Longueur minimale d'un document en caractères.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=FineWebConfig.progress_interval,
        help="Fréquence de mise à jour (en documents conservés).",
    )
    parser.add_argument(
        "--lang-detect-max-chars",
        type=int,
        default=FineWebConfig.lang_detect_max_chars,
        help="Longueur max utilisée par langdetect (limite la latence sur gros documents).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    max_bytes = None
    if args.max_mib is not None:
        max_bytes = int(args.max_mib * 1024 * 1024)

    return FineWebConfig(
        dataset_path=args.dataset_path,
        config_name=args.config_name,
        split=args.split,
        output_dir=args.output_dir,
        max_docs=max(0, args.max_docs),
        max_bytes=max_bytes,
        lang_threshold=args.lang_threshold,
        min_chars=max(0, args.min_chars),
        progress_interval=max(1, args.progress_interval),
        lang_detect_max_chars=max(0, args.lang_detect_max_chars),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        cfg = parse_args(argv)
        summary = download_fineweb_fr(cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
