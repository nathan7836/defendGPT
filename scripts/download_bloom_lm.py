#!/usr/bin/env python3
"""Download selected language splits from the sil-ai/bloom-lm dataset."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class BloomConfig:
    dataset: str = "sil-ai/bloom-lm"
    languages: list[str] = field(default_factory=lambda: ["eng"])
    splits: list[str] = field(default_factory=lambda: ["train", "validation", "test"])
    output_dir: Path = Path("data_clean") / "bloom_lm"
    max_docs: Optional[int] = None
    progress_interval: int = 200


def _ensure_datasets() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Le paquet 'datasets' est requis. Installez-le avec 'pip install datasets'."
        ) from exc


def _stream_split(cfg: BloomConfig, language: str, split: str):
    from datasets import load_dataset

    return load_dataset(cfg.dataset, language, split=split, streaming=True)


def download_bloom(cfg: Optional[BloomConfig] = None) -> list[dict[str, object]]:
    _ensure_datasets()

    cfg = cfg or BloomConfig()
    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    sentinel = "\n\n<|doc|>\n\n"

    for language in cfg.languages:
        language = language.strip()
        if not language:
            continue
        for split in cfg.splits:
            split = split.strip()
            if not split:
                continue

            output_path = cfg.output_dir / f"bloom_lm_{language}_{split}.txt"
            metadata_path = output_path.with_suffix(".meta.json")

            processed = 0
            kept = 0
            bytes_written = 0
            start_ts = time.time()
            handle = None

            try:
                try:
                    dataset_iter = _stream_split(cfg, language, split)
                except Exception as exc:  # noqa: BLE001
                    print(f"[bloom] skip {language}/{split}: {exc}")
                    continue

                for example in dataset_iter:
                    processed += 1
                    text = example.get("text") if isinstance(example, dict) else None
                    if not text:
                        continue
                    text = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
                    if not text:
                        continue

                    if handle is None:
                        handle = output_path.open("w", encoding="utf-8")
                    if kept > 0:
                        handle.write(sentinel)
                    handle.write(text)
                    kept += 1
                    bytes_written = handle.tell()

                    if cfg.max_docs and kept >= cfg.max_docs:
                        break

                    if kept % max(1, cfg.progress_interval) == 0:
                        print(
                            f"[bloom] {language}/{split} kept={kept} processed={processed} "
                            f"bytes={bytes_written/1e6:.2f}MB",
                            flush=True,
                        )

            finally:
                if handle is not None:
                    handle.close()

            if kept == 0:
                if output_path.exists():
                    output_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                continue

            summary = {
                "dataset": cfg.dataset,
                "language": language,
                "split": split,
                "processed": processed,
                "kept_docs": kept,
                "bytes_written": bytes_written,
                "elapsed_seconds": round(time.time() - start_ts, 2),
                "output_path": str(output_path),
                "max_docs": cfg.max_docs,
            }
            summaries.append(summary)

            with metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)

            print(
                f"[bloom] completed {language}/{split}: kept={kept} processed={processed} "
                f"bytes={bytes_written/1e6:.2f}MB",
                flush=True,
            )

    return summaries


def _all_language_codes() -> list[str]:
    _ensure_datasets()
    from datasets import load_dataset_builder

    builder = load_dataset_builder("sil-ai/bloom-lm")
    return sorted(builder.builder_configs.keys())


def parse_args(argv: Optional[Iterable[str]] = None) -> BloomConfig:
    parser = argparse.ArgumentParser(description="Télécharger des sous-corpus sil-ai/bloom-lm")
    parser.add_argument("--dataset", default=BloomConfig.dataset)
    parser.add_argument(
        "--language",
        "-l",
        action="append",
        help="Code langue (ISO 639-3). Répétez l'option pour plusieurs langues.",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Télécharger toutes les langues disponibles (attention au volume).",
    )
    parser.add_argument(
        "--splits",
        default=",".join(BloomConfig().splits),
        help="Liste de splits séparés par des virgules (ex: train,validation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BloomConfig.output_dir,
        help="Répertoire cible (défaut: data_clean/bloom_lm)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limiter le nombre de documents par split (optionnel).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=BloomConfig.progress_interval,
        help="Fréquence d'affichage des logs (en documents conservés)",
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="Lister les codes de langue disponibles puis quitter.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_languages:
        codes = _all_language_codes()
        print("\n".join(codes))
        raise SystemExit(0)

    if args.all_languages:
        languages = _all_language_codes()
    elif args.language:
        languages = args.language
    else:
        languages = BloomConfig().languages

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise SystemExit("Aucun split fourni.")

    return BloomConfig(
        dataset=args.dataset,
        languages=[lang.strip() for lang in languages if lang.strip()],
        splits=splits,
        output_dir=args.output_dir,
        max_docs=args.max_docs if (args.max_docs or 0) > 0 else None,
        progress_interval=max(1, args.progress_interval),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        cfg = parse_args(argv)
        summaries = download_bloom(cfg)
    except KeyboardInterrupt:
        print("Interrompu par l'utilisateur.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

