#!/usr/bin/env python3
"""Train a subword tokenizer (BPE) on a text corpus."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFD, Sequence, StripAccents
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data_clean"
DEFAULT_OUTPUT_DIR = ROOT / "trained_models" / "tokenizers"


@dataclass
class TokenizerConfig:
    name: str
    input_dir: Path
    output_dir: Path
    vocab_size: int = 32_000
    min_frequency: int = 2
    lowercase: bool = False
    limit_files: Optional[int] = None
    limit_bytes: Optional[int] = None
    special_tokens: Optional[List[str]] = None


def _iter_files(directory: Path, limit: Optional[int] = None) -> List[Path]:
    files = [path for path in sorted(directory.rglob("*.txt")) if path.is_file()]
    if limit is not None:
        return files[:limit]
    return files


def _read_length(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def train_tokenizer(
    cfg: TokenizerConfig,
    progress_cb: Optional[Callable[[Dict[str, int | str]], None]] = None,
) -> Dict[str, int | str]:
    output_dir = cfg.output_dir / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_files(cfg.input_dir, cfg.limit_files)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {cfg.input_dir}")

    if cfg.limit_bytes is not None:
        running = 0
        limited_files = []
        for path in files:
            running += _read_length(path)
            limited_files.append(path)
            if running >= cfg.limit_bytes:
                break
        files = limited_files

    if progress_cb:
        progress_cb({"event": "initialised", "total_files": len(files)})

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    normalizers_list = [NFD(), StripAccents()]
    if cfg.lowercase:
        normalizers_list.append(Lowercase())
    tokenizer.normalizer = Sequence(normalizers_list)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    special_tokens = cfg.special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )

    start = time.time()
    tokenizer.train(files=[str(path) for path in files], trainer=trainer)

    vocab = tokenizer.get_vocab()
    cls_id = vocab.get("[CLS]", 0)
    sep_id = vocab.get("[SEP]", 0)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    duration = int(time.time() - start)

    output_path = output_dir / "tokenizer.json"
    tokenizer.save(str(output_path))

    metadata = {
        "name": cfg.name,
        "input_dir": str(cfg.input_dir),
        "output_dir": str(output_dir),
        "vocab_size": cfg.vocab_size,
        "min_frequency": cfg.min_frequency,
        "lowercase": cfg.lowercase,
        "special_tokens": special_tokens,
        "files_used": len(files),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": duration,
        "tokenizer_path": str(output_path),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if progress_cb:
        progress_cb({"event": "completed", "tokenizer_path": str(output_path)})

    return metadata


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a corpus")
    parser.add_argument("--name", default="wiki-bpe")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-mb", type=float, default=None, help="Limit corpus size in MiB")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before training")
    parser.add_argument("--stats", type=Path, default=None, help="Optional JSON output with training metadata")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    limit_bytes = None
    if args.limit_mb is not None:
        limit_bytes = math.floor(args.limit_mb * 1024 * 1024)

    cfg = TokenizerConfig(
        name=args.name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=args.lowercase,
        limit_files=args.limit_files,
        limit_bytes=limit_bytes,
    )
    summary = train_tokenizer(cfg)
    if args.stats:
        args.stats.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
