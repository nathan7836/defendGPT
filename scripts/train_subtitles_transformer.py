#!/usr/bin/env python3
"""Train a tiny Transformer language model on the cleaned subtitle corpus."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Set

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data_clean"

# Predefined architecture presets inspired by common GPT size tiers.
MODEL_ARCH_PRESETS: dict[str, dict[str, int | float]] = {
    "very-small": {
        "num_layers": 4,
        "num_heads": 4,
        "embed_dim": 192,
        "ff_hidden_dim": 768,
        "block_size": 512,
        "vocab_size": 16_000,
        "batch_size": 16,
        "max_steps": 300,
        "lr": 3e-4,
        "eval_interval": 50,
        "eval_batches": 8,
        "metrics_log_fraction": 0.05,
        "sample_interval": 150,
        "sample_max_new_tokens": 80,
        "sample_temperature": 0.8,
    },
    "small": {
        "num_layers": 8,
        "num_heads": 8,
        "embed_dim": 512,
        "ff_hidden_dim": 2_048,
        "block_size": 768,
        "vocab_size": 32_000,
        "batch_size": 8,
        "max_steps": 600,
        "lr": 2.5e-4,
        "eval_interval": 75,
        "eval_batches": 12,
        "metrics_log_fraction": 0.04,
        "sample_interval": 200,
        "sample_max_new_tokens": 100,
        "sample_temperature": 0.85,
    },
    "medium": {
        "num_layers": 18,
        "num_heads": 16,
        "embed_dim": 1_024,
        "ff_hidden_dim": 4_096,
        "block_size": 1_024,
        "vocab_size": 32_000,
        "batch_size": 4,
        "max_steps": 1_000,
        "lr": 2e-4,
        "eval_interval": 100,
        "eval_batches": 16,
        "metrics_log_fraction": 0.05,
        "sample_interval": 250,
        "sample_max_new_tokens": 120,
        "sample_temperature": 0.9,
    },
}

# Backward compat aliases for legacy preset names
MODEL_ARCH_PRESETS["mini-gpt"] = MODEL_ARCH_PRESETS["very-small"].copy()
MODEL_ARCH_PRESETS["small-gpt"] = MODEL_ARCH_PRESETS["small"].copy()
MODEL_ARCH_PRESETS["medium-4gb"] = MODEL_ARCH_PRESETS["medium"].copy()


def optional_path(value: str) -> Optional[Path]:
    lowered = value.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return Path(value)


def config_to_dict(cfg: Config) -> dict[str, object]:
    data: dict[str, object] = {}
    for key, value in cfg.__dict__.items():
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, list):
            data[key] = [str(item) if isinstance(item, Path) else item for item in value]
        elif isinstance(value, tuple):
            data[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            data[key] = value
    return data


def apply_arch_preset(cfg: Config, preset_name: str) -> None:
    preset_key = preset_name.lower()
    preset = MODEL_ARCH_PRESETS.get(preset_key)
    if preset is None:
        available = ", ".join(sorted(MODEL_ARCH_PRESETS)) or "aucun"
        raise ValueError(f"Preset inconnu '{preset_name}'. Options: {available}")
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.arch_preset = preset_key


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device(request: str) -> torch.device:
    if request != "auto":
        return torch.device(request)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class Config:
    data_dir: Path = DEFAULT_DATA_DIR
    block_size: int = 256
    min_seq_len: int = 0
    batch_size: int = 16
    embed_dim: int = 128
    vocab_size: int = 256
    layernorm_dim: Optional[int] = None
    head_dim: Optional[int] = None
    num_heads: int = 4
    num_layers: int = 4
    ff_hidden_dim: int = 512
    dropout: float = 0.1
    max_steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.0
    eval_interval: int = 50
    eval_batches: int = 10
    train_split: float = 0.95
    seed: int = 13
    device: str = "auto"
    log_dir: Optional[Path] = ROOT / "trained_models" / "runs"
    run_name: Optional[str] = None
    sample_interval: int = 100
    sample_prompt: str = "Bonjour je suis "
    sample_max_new_tokens: int = 60
    sample_temperature: float = 0.8
    sample_top_k: int = 50
    metrics_log_fraction: float = 0.05
    arch_preset: Optional[str] = None
    extra_data_dirs: list[Path] = field(default_factory=list)
    resume_checkpoint: Optional[Path] = None
    resume_run_dir: Optional[Path] = None


def _coerce_value(example: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(example, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(example, int) and not isinstance(example, bool):
        return int(value)
    if isinstance(example, float):
        return float(value)
    return value


def config_from_dict(payload: Dict[str, Any], base: Optional[Config] = None) -> Config:
    """Create a Config instance from a JSON-like payload."""

    cfg = base or Config()
    for key, value in payload.items():
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        if key in {"layernorm_dim", "head_dim"}:
            if value in {None, "", "none", "null"}:
                setattr(cfg, key, None)
            else:
                try:
                    setattr(cfg, key, int(value))
                except (TypeError, ValueError):
                    continue
            continue
        if isinstance(current, Path):
            if value in {None, "", "none", "null"}:
                setattr(cfg, key, None)
            else:
                setattr(cfg, key, Path(value))
        elif isinstance(current, list):
            if isinstance(value, str):
                candidates = [segment.strip() for segment in value.split(",") if segment.strip()]
            elif isinstance(value, (list, tuple, set)):
                candidates = [str(item) for item in value if str(item).strip()]
            elif value in {None, "", "none", "null"}:
                candidates = []
            else:
                continue
            paths: list[Path] = []
            for candidate in candidates:
                try:
                    paths.append(Path(candidate))
                except TypeError:
                    continue
            setattr(cfg, key, paths)
        else:
            try:
                coerced = _coerce_value(current, value)
            except (TypeError, ValueError):
                continue
            setattr(cfg, key, coerced)
    return cfg


class MetricsLogger:
    def __init__(self, run_dir: Optional[Path]) -> None:
        self.run_dir = run_dir
        self._file = None
        self.latest_path = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = run_dir / "metrics.jsonl"
            self._file = open(self.metrics_path, "a", encoding="utf-8", buffering=1)
            self.latest_path = run_dir / "latest.json"

    def log(self, step: int, split: str, loss: float) -> None:
        if self._file is None:
            return
        record = {
            "timestamp": time.time(),
            "step": int(step),
            "split": split,
            "loss": float(loss),
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self.latest_path is not None:
            with open(self.latest_path, "w", encoding="utf-8") as latest:
                json.dump(record, latest)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


class SampleLogger:
    def __init__(self, run_dir: Optional[Path]) -> None:
        self._file = None
        self.path: Optional[Path] = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.path = run_dir / "samples.txt"
            self._file = open(self.path, "a", encoding="utf-8", buffering=1)

    def log(self, step: int, sample: str) -> None:
        if self._file is None:
            return
        flattened = sample.replace("\r\n", " ").replace("\n", " ")
        self._file.write(f"step {step:04d}: {flattened}\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _format_token_display(token_id: int) -> str:
    if 0 <= token_id <= 255:
        char = chr(token_id)
        special_map = {
            " ": "␠",
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            "\v": "\\v",
            "\f": "\\f",
        }
        if char in special_map:
            return special_map[char]
        if char.isprintable():
            return char
        return f"0x{token_id:02x}"
    return f"#{token_id}"


def _pca_projection(
    embeddings: torch.Tensor, n_components: int = 3
) -> tuple[list[list[float]], list[float]]:
    """Compute PCA coordinates and variance ratios for embedding weights."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings tensor must be 2D")
    centered = embeddings.detach().to(torch.float32)
    centered -= centered.mean(dim=0, keepdim=True)
    samples, dims = centered.shape
    comps = min(n_components, dims, samples)
    if comps == 0:
        return [], []
    if samples < 2:
        coords = centered[:, :comps].cpu().tolist()
        return coords, [1.0] + [0.0] * (comps - 1)

    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:comps, :].transpose(0, 1)
    projection = centered @ components
    denom = max(samples - 1, 1)
    variances = (s[:comps] ** 2) / denom
    total_var = (s**2).sum() / denom
    if total_var.item() == 0:
        ratios = [0.0 for _ in range(comps)]
    else:
        ratios = (variances / total_var).cpu().tolist()
    coords = projection[:, :comps].cpu().tolist()
    return coords, ratios


class VisualLogger:
    """Append JSONL snapshots for embedding PCA and logits summaries."""

    def __init__(self, run_dir: Optional[Path], top_k: int = 8) -> None:
        self.top_k = top_k
        self._file = None
        self.latest_path: Optional[Path] = None
        self.path: Optional[Path] = None
        if run_dir is not None:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.path = run_dir / "visuals.jsonl"
            self._file = open(self.path, "a", encoding="utf-8", buffering=1)
            self.latest_path = run_dir / "visuals_latest.json"

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def log(
        self,
        step: int,
        embeddings: torch.Tensor,
        tail_logits: Optional[torch.Tensor],
        *,
        sample: Optional[str] = None,
    ) -> None:
        if self._file is None:
            return
        coords, variance_ratio = _pca_projection(embeddings, n_components=3)
        record: dict[str, object] = {
            "timestamp": time.time(),
            "step": int(step),
            "embedding": {
                "token_ids": list(range(len(coords))),
                "coords": coords,
                "variance_ratio": variance_ratio,
                "token_strings": [_format_token_display(token_id) for token_id in range(len(coords))],
            },
        }

        if tail_logits is not None:
            probs = torch.softmax(tail_logits.detach().to(torch.float32), dim=0)
            top_k = min(self.top_k, probs.numel())
            top_values, top_indices = torch.topk(probs, top_k)
            entropy = float(
                -(probs * torch.clamp(probs, min=1e-9).log()).sum().item()
            )
            record["logits"] = {
                "token_ids": top_indices.cpu().tolist(),
                "probabilities": [float(v.item()) for v in top_values],
                "entropy": entropy,
                "max_probability": float(top_values[0].item()) if top_values.numel() else None,
            }

        if sample:
            record["sample"] = sample

        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        if self.latest_path is not None:
            with open(self.latest_path, "w", encoding="utf-8") as latest:
                json.dump(record, latest)


class SubtitleCorpus:
    def __init__(self, directory: Path, extra_dirs: Optional[Sequence[Path]] = None) -> None:
        primary = Path(directory)
        directories: list[Path] = [primary]
        if extra_dirs:
            for extra in extra_dirs:
                try:
                    extra_path = Path(extra)
                except TypeError:
                    continue
                if extra_path not in directories:
                    directories.append(extra_path)
        self.directories = directories

    def _iter_files(self) -> list[Path]:
        files: list[Path] = []
        for directory in self.directories:
            if not directory.exists():
                print(f"[warn] Corpus directory missing: {directory}")
                continue
            files.extend(sorted(directory.rglob("*.txt")))
        # Deduplicate while preserving order
        seen: Set[Path] = set()
        unique_files: list[Path] = []
        for path in files:
            if path in seen:
                continue
            seen.add(path)
            unique_files.append(path)
        return unique_files

    def load_documents(self) -> list[str]:
        files = self._iter_files()
        if not files:
            raise FileNotFoundError(
                "No .txt files found in configured directories: "
                + ", ".join(str(path) for path in self.directories)
            )
        docs: list[str] = []
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(text)
        if not docs:
            raise RuntimeError("Corpus directories contain only blank files after filtering.")
        return docs

    def to_bytes(self) -> bytes:
        sentinel = "\n\n<|doc|>\n\n"
        docs = self.load_documents()
        return sentinel.join(docs).encode("utf-8")

    def to_tensor(self) -> torch.Tensor:
        corpus_bytes = self.to_bytes()
        return torch.tensor(list(corpus_bytes), dtype=torch.long)


class ByteDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int, min_seq_len: int) -> None:
        if data.ndim != 1:
            raise ValueError("data must be a 1D tensor")
        if len(data) <= block_size:
            raise ValueError("data is shorter than the configured block size")
        self.data = data
        self.block_size = block_size
        if min_seq_len <= 0:
            raise ValueError("min_seq_len must be > 0")
        if min_seq_len > block_size:
            raise ValueError("min_seq_len cannot exceed block_size")
        self.min_seq_len = min_seq_len

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        max_len = chunk.size(0) - 1
        if max_len < self.min_seq_len:
            raise ValueError("Chunk shorter than minimum sequence length")
        if self.min_seq_len == max_len:
            seq_len = max_len
        else:
            seq_len = random.randint(self.min_seq_len, max_len)
        src = chunk[:seq_len]
        tgt = chunk[1 : seq_len + 1]
        return {"src": src, "tgt": tgt, "length": torch.tensor(seq_len, dtype=torch.long)}


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "pos_encoding",
            self._create_encoding(embed_dim, max_seq_len),
            persistent=False,
        )

    @staticmethod
    def _create_encoding(embed_dim: int, max_seq_len: int) -> torch.Tensor:
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        encoding = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len]


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        vocab_size = cfg.vocab_size
        self.tok_embed = nn.Embedding(vocab_size, cfg.embed_dim)
        self.pos_encoding = PositionalEncoding(cfg.embed_dim, cfg.block_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        ln_dim = cfg.layernorm_dim or cfg.embed_dim
        head_dim = cfg.head_dim or ln_dim

        self.pre_ln_proj: nn.Linear | None = None
        if ln_dim != cfg.embed_dim:
            self.pre_ln_proj = nn.Linear(cfg.embed_dim, ln_dim)

        self.ln = nn.LayerNorm(ln_dim)

        self.head_pre: nn.Linear | None = None
        if head_dim != ln_dim:
            self.head_pre = nn.Linear(ln_dim, head_dim)

        self.head = nn.Linear(head_dim, vocab_size, bias=False)
        if self.pre_ln_proj is None and self.head_pre is None and head_dim == cfg.embed_dim:
            # Weight tying only possible when dimensions align perfectly.
            self.head.weight = self.tok_embed.weight
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(
        self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # tokens: (batch, seq_len)
        x = self.tok_embed(tokens)
        x = self.pos_encoding(x)
        seq_len = tokens.size(1)
        mask = self.causal_mask[:seq_len, :seq_len].to(tokens.device)
        if padding_mask is not None:
            padding_mask = padding_mask[:, :seq_len]
            padding_mask = padding_mask.to(tokens.device)
        x = self.encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        if self.pre_ln_proj is not None:
            x = self.pre_ln_proj(x)
        x = self.ln(x)
        if self.head_pre is not None:
            x = self.head_pre(x)
        return self.head(x)


class SubtitleDataModule:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        normalised_extras: list[Path] = []
        for entry in cfg.extra_data_dirs:
            try:
                normalised_extras.append(Path(entry))
            except TypeError:
                continue
        self.cfg.extra_data_dirs = normalised_extras
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._min_seq_len = self._resolve_min_seq_len()
        self.cfg.min_seq_len = self._min_seq_len

    def _resolve_min_seq_len(self) -> int:
        if self.cfg.min_seq_len > 0:
            candidate = self.cfg.min_seq_len
        else:
            candidate = max(1, self.cfg.block_size // 2)
        if candidate > self.cfg.block_size:
            print(
                f"[warn] min_seq_len ({candidate}) clipped to block_size ({self.cfg.block_size})"
            )
        return max(1, min(candidate, self.cfg.block_size))

    @staticmethod
    def _collate_batch(samples: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("Batch is empty")
        max_len = max(int(sample["length"]) for sample in samples)
        batch_size = len(samples)
        tokens = torch.zeros(batch_size, max_len, dtype=torch.long)
        targets = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for row, sample in enumerate(samples):
            seq_len = int(sample["length"])
            tokens[row, :seq_len] = sample["src"][:seq_len]
            targets[row, :seq_len] = sample["tgt"][:seq_len]
            padding_mask[row, :seq_len] = False
        return tokens, targets, padding_mask

    def setup(self) -> None:
        if self._train_loader is not None and self._val_loader is not None:
            return
        corpus = SubtitleCorpus(self.cfg.data_dir, extra_dirs=self.cfg.extra_data_dirs)
        data_tensor = corpus.to_tensor()
        split_idx = int(len(data_tensor) * self.cfg.train_split)
        train_tensor = data_tensor[:split_idx]
        val_tensor = data_tensor[split_idx:]
        if len(val_tensor) <= self.cfg.block_size:
            val_tensor = train_tensor[-(self.cfg.block_size + len(val_tensor) + 1) :]

        train_ds = ByteDataset(train_tensor, self.cfg.block_size, self._min_seq_len)
        val_ds = ByteDataset(val_tensor, self.cfg.block_size, self._min_seq_len)

        self._train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_batch,
        )
        self._val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self._collate_batch,
        )

    @property
    def train_loader(self) -> DataLoader:
        self.setup()
        assert self._train_loader is not None
        return self._train_loader

    @property
    def val_loader(self) -> DataLoader:
        self.setup()
        assert self._val_loader is not None
        return self._val_loader


def _param_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def summarise_model(model: TinyTransformerLM) -> dict[str, object]:
    total_params, trainable_params = _param_counts(model)
    embedding_total, embedding_trainable = _param_counts(model.tok_embed)
    ln_total, ln_trainable = _param_counts(model.ln)
    if model.pre_ln_proj is not None:
        proj_total, proj_trainable = _param_counts(model.pre_ln_proj)
        ln_total += proj_total
        ln_trainable += proj_trainable
    head_total, head_trainable = _param_counts(model.head)
    if model.head_pre is not None:
        head_pre_total, head_pre_trainable = _param_counts(model.head_pre)
        head_total += head_pre_total
        head_trainable += head_pre_trainable

    layers_summary: list[dict[str, object]] = []
    if hasattr(model.encoder, "layers"):
        for idx, layer in enumerate(model.encoder.layers):  # type: ignore[attr-defined]
            layer_total, layer_trainable = _param_counts(layer)
            layers_summary.append(
                {
                    "name": f"Bloc {idx + 1}",
                    "params": layer_total,
                    "trainable": layer_trainable,
                }
            )

    ln_shape = getattr(model.ln, "normalized_shape", ())
    if isinstance(ln_shape, torch.Size):
        ln_dim = int(ln_shape[0]) if len(ln_shape) else model.cfg.embed_dim
    elif isinstance(ln_shape, (tuple, list)):
        ln_dim = int(ln_shape[0]) if ln_shape else model.cfg.embed_dim
    else:
        ln_dim = model.cfg.embed_dim

    head_dim = getattr(model.head, "in_features", model.cfg.embed_dim)

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "blocks": [
            {
                "name": "Embedding",
                "params": embedding_total,
                "trainable": embedding_trainable,
            },
            {
                "name": "LayerNorm",
                "params": ln_total,
                "trainable": ln_trainable,
            },
            {
                "name": "Head",
                "params": head_total,
                "trainable": head_trainable,
            },
        ],
        "encoder_layers": layers_summary,
        "config": {
            "arch_preset": model.cfg.arch_preset,
            "embed_dim": model.cfg.embed_dim,
            "layernorm_dim": model.cfg.layernorm_dim or ln_dim,
            "num_layers": model.cfg.num_layers,
            "num_heads": model.cfg.num_heads,
            "ff_hidden_dim": model.cfg.ff_hidden_dim,
            "block_size": model.cfg.block_size,
            "dropout": model.cfg.dropout,
            "head_dim": model.cfg.head_dim or head_dim,
            "vocab_size": model.cfg.vocab_size,
        },
    }
    return summary


class SubtitleTrainer:
    def __init__(
        self,
        cfg: Config,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = auto_device(cfg.device)
        print(f"Using device: {self.device}")

        if self.cfg.resume_run_dir is not None and not self.cfg.run_name:
            self.cfg.run_name = Path(self.cfg.resume_run_dir).name
        if self.cfg.resume_run_dir is not None and self.cfg.log_dir is None:
            self.cfg.log_dir = Path(self.cfg.resume_run_dir).parent

        self.gpu_info = None
        if self.device.type == "cuda":
            gpu_index = (
                self.device.index if self.device.index is not None else torch.cuda.current_device()
            )
            props = torch.cuda.get_device_properties(gpu_index)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(
                f"GPU: {props.name} | compute capability {props.major}.{props.minor} |"
                f" {total_mem_gb:.2f} GB total memory"
            )
            self.gpu_info = {
                "index": gpu_index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
            }

        self.data_module = SubtitleDataModule(cfg)
        self.model = TinyTransformerLM(cfg).to(self.device)
        self.model_summary = summarise_model(self.model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        self.resume_checkpoint_path: Optional[Path] = None
        if self.cfg.resume_checkpoint is not None:
            self.resume_checkpoint_path = Path(self.cfg.resume_checkpoint)
        elif self.cfg.resume_run_dir is not None:
            candidate = Path(self.cfg.resume_run_dir) / "checkpoint.pt"
            if candidate.exists():
                self.resume_checkpoint_path = candidate

        self.start_step = 0
        if self.resume_checkpoint_path is not None and self.resume_checkpoint_path.exists():
            print(f"[resume] Loading checkpoint from {self.resume_checkpoint_path}")
            checkpoint = torch.load(self.resume_checkpoint_path, map_location=self.device)
            model_state = checkpoint.get("model_state")
            if model_state is None:
                raise ValueError("Checkpoint does not contain model_state")
            self.model.load_state_dict(model_state)
            optim_state = checkpoint.get("optimizer_state")
            if optim_state is not None:
                try:
                    self.optimizer.load_state_dict(optim_state)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] Impossible de charger l'état de l'optimiseur: {exc}")
            self.start_step = int(checkpoint.get("step", 0))
            print(f"[resume] Reprise à partir de l'étape {self.start_step}")
        elif self.resume_checkpoint_path is not None:
            print(f"[warn] Checkpoint introuvable: {self.resume_checkpoint_path}")
            self.resume_checkpoint_path = None

        self.metadata: dict[str, object] = {}
        self.metadata_path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self.run_checkpoint_path: Optional[Path] = None
        self.metrics_logger: Optional[MetricsLogger] = None
        self.sample_logger: Optional[SampleLogger] = None
        self.visual_logger: Optional[VisualLogger] = None
        self.status = "completed"
        self.train_log_interval = max(1, int(max(1, self.cfg.max_steps) * max(0.0, self.cfg.metrics_log_fraction)))
        self.progress_callback = progress_callback
        self.stop_event = stop_event

        self.checkpoint_dir = ROOT / "trained_models"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_path = self.checkpoint_dir / "tiny_subtitles_transformer.pt"

    def _emit_progress(self, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            pass

    def _should_stop(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def _prepare_logging(self) -> None:
        if self.cfg.resume_run_dir is not None:
            run_dir = Path(self.cfg.resume_run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self.run_dir = run_dir
            self.run_checkpoint_path = run_dir / "checkpoint.pt"
            self.metadata_path = run_dir / "metadata.json"
            existing: dict[str, object] = {}
            if self.metadata_path.exists():
                try:
                    existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
            resumed_at = datetime.now().isoformat()
            existing.update(
                {
                    "run_name": self.cfg.run_name or run_dir.name,
                    "resumed_at": resumed_at,
                    "device": str(self.device),
                    "gpu": self.gpu_info,
                    "config": config_to_dict(self.cfg),
                    "status": "running",
                    "metrics_file": str(run_dir / "metrics.jsonl"),
                    "samples_file": str(run_dir / "samples.txt"),
                    "model_summary": self.model_summary,
                }
            )
            self.metadata = existing
            with open(self.metadata_path, "w", encoding="utf-8") as meta_file:
                json.dump(self.metadata, meta_file, indent=2)
            print(f"Resuming logging in {self.run_dir}")
            self.metrics_logger = MetricsLogger(run_dir)
            self.sample_logger = SampleLogger(run_dir)
            self.visual_logger = VisualLogger(run_dir)
            self._emit_progress(
                event="run_resumed",
                run_dir=str(run_dir),
                run_name=self.metadata.get("run_name"),
                metadata_path=str(self.metadata_path),
            )
            return

        if self.cfg.log_dir is None:
            self.metrics_logger = MetricsLogger(None)
            self.sample_logger = SampleLogger(None)
            self.visual_logger = VisualLogger(None)
            self._emit_progress(event="run_created", run_dir=None, run_name=self.cfg.run_name)
            return
        log_root = Path(self.cfg.log_dir)
        run_name = self.cfg.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        self.cfg.run_name = run_name
        self.run_dir = log_root / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_checkpoint_path = self.run_dir / "checkpoint.pt"

        stale_files = [
            self.run_dir / "metrics.jsonl",
            self.run_dir / "latest.json",
            self.run_dir / "visuals.jsonl",
            self.run_dir / "visuals_latest.json",
            self.run_dir / "samples.txt",
            self.run_checkpoint_path,
        ]
        for path in stale_files:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

        self.metadata = {
            "run_name": run_name,
            "started_at": datetime.now().isoformat(),
            "device": str(self.device),
            "gpu": self.gpu_info,
            "config": config_to_dict(self.cfg),
            "status": "running",
            "metrics_file": str(self.run_dir / "metrics.jsonl"),
            "samples_file": str(self.run_dir / "samples.txt"),
            "model_summary": self.model_summary,
        }
        self.metadata_path = self.run_dir / "metadata.json"
        with open(self.metadata_path, "w", encoding="utf-8") as meta_file:
            json.dump(self.metadata, meta_file, indent=2)
        print(f"Logging metrics to {self.run_dir}")
        self.metrics_logger = MetricsLogger(self.run_dir)
        self.sample_logger = SampleLogger(self.run_dir)
        self.visual_logger = VisualLogger(self.run_dir)
        self._emit_progress(
            event="run_created",
            run_dir=str(self.run_dir),
            run_name=run_name,
            metadata_path=str(self.metadata_path),
        )

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        seen = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                src, tgt, padding_mask = batch
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                padding_mask = padding_mask.to(self.device)
                logits = self.model(src, padding_mask=padding_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                total_loss += loss.item()
                seen += 1
                if batch_idx + 1 >= self.cfg.eval_batches:
                    break
        self.model.train()
        return total_loss / max(seen, 1)

    def sample_text(self) -> Optional[str]:
        if self.cfg.sample_interval <= 0:
            return None
        prompt = self.cfg.sample_prompt or ""
        input_bytes = prompt.encode("utf-8") or b"\n"
        tokens = torch.tensor(list(input_bytes), dtype=torch.long, device=self.device).unsqueeze(0)
        generated: list[int] = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for _ in range(max(self.cfg.sample_max_new_tokens, 0)):
                window = tokens[:, -self.cfg.block_size :]
                logits = self.model(window)
                logits = logits[:, -1, :] / max(self.cfg.sample_temperature, 1e-5)
                if 0 < self.cfg.sample_top_k < logits.size(-1):
                    values, _ = torch.topk(logits, self.cfg.sample_top_k, dim=-1)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < cutoff, torch.full_like(logits, float("-inf")), logits
                    )
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
                generated.append(int(next_token.item()))
        if was_training:
            self.model.train()
        new_text = bytes(generated).decode("utf-8", errors="ignore")
        full_text = (input_bytes + bytes(generated)).decode("utf-8", errors="ignore")
        return full_text

    def _finalise_metadata(self, completed_steps: int) -> None:
        if self.metadata_path is None:
            return
        payload = {
            "status": self.status,
            "ended_at": datetime.now().isoformat(),
            "completed_steps": completed_steps,
            "final_checkpoint": str(self.save_path) if self.status == "completed" else None,
            "run_checkpoint": str(self.run_checkpoint_path)
            if (self.run_checkpoint_path and self.status == "completed")
            else None,
            "visuals_file": str(self.visual_logger.path)
            if (self.visual_logger and self.visual_logger.path is not None)
            else None,
        }
        self.metadata.update(payload)
        with open(self.metadata_path, "w", encoding="utf-8") as meta_file:
            json.dump(self.metadata, meta_file, indent=2)

    def run(self) -> None:
        self._prepare_logging()
        assert self.metrics_logger is not None

        train_loader = self.data_module.train_loader
        val_loader = self.data_module.val_loader

        step = self.start_step
        if step >= self.cfg.max_steps:
            print(
                f"[train] max_steps ({self.cfg.max_steps}) déjà atteints, aucune nouvelle étape exécutée."
            )
        if self.resume_checkpoint_path is not None and step < self.cfg.max_steps:
            print(f"[resume] Poursuite de l'entraînement jusqu'à {self.cfg.max_steps} étapes")
        train_iter = iter(train_loader)

        try:
            while step < self.cfg.max_steps:
                if self._should_stop():
                    print("[train] Stop signal received, finalising run…")
                    self.status = "stopped"
                    break
                try:
                    src, tgt, padding_mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    src, tgt, padding_mask = next(train_iter)

                src = src.to(self.device)
                tgt = tgt.to(self.device)
                padding_mask = padding_mask.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(src, padding_mask=padding_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                step += 1
                train_loss = loss.item()
                should_log_train = (
                    step == self.start_step + 1
                    or step == self.cfg.max_steps
                    or step % self.train_log_interval == 0
                )

                sample_text: Optional[str] = None
                if self.cfg.sample_interval > 0 and step % self.cfg.sample_interval == 0:
                    sample_text = self.sample_text()
                    if sample_text is not None:
                        if self.sample_logger is not None:
                            self.sample_logger.log(step, sample_text)
                        preview = sample_text.replace("\n", " ")
                        if len(preview) > 200:
                            preview = preview[:197] + "..."
                        print(f"step {step:04d}: {preview}")
                if should_log_train:
                    self.metrics_logger.log(step, "train", train_loss)
                    print(f"step={step:04d} train_loss={train_loss:.4f}")
                    self._emit_progress(
                        event="train",
                        step=step,
                        train_loss=train_loss,
                        run_dir=str(self.run_dir) if self.run_dir else None,
                    )

                if self.visual_logger is not None and (should_log_train or sample_text is not None):
                    embeddings_snapshot = self.model.tok_embed.weight.detach().cpu()
                    valid_positions = (~padding_mask[0]).nonzero(as_tuple=False)
                    tail_index = int(valid_positions[-1]) if valid_positions.numel() else -1
                    tail_logits = logits.detach()[0, tail_index, :].cpu()
                    self.visual_logger.log(
                        step,
                        embeddings_snapshot,
                        tail_logits,
                        sample=sample_text,
                    )

                if self._should_stop():
                    print("[train] Stop signal received before evaluation.")
                    self.status = "stopped"
                    break

                if step % self.cfg.eval_interval == 0 or step == self.cfg.max_steps:
                    val_loss = self.evaluate(val_loader)
                    self.metrics_logger.log(step, "val", val_loss)
                    print(f"step={step:04d} val_loss={val_loss:.4f}")
                    self._emit_progress(
                        event="val",
                        step=step,
                        val_loss=val_loss,
                        run_dir=str(self.run_dir) if self.run_dir else None,
                    )

                if self._should_stop():
                    print("[train] Stop signal received after sampling.")
                    self.status = "stopped"
                    break

            checkpoint = {
                "config": self.cfg.__dict__,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": step,
            }
            torch.save(checkpoint, self.save_path)
            if self.run_checkpoint_path is not None:
                torch.save(checkpoint, self.run_checkpoint_path)
            print(f"Saved checkpoint to {self.save_path}")
            if self.run_checkpoint_path is not None:
                print(f"Run-specific checkpoint: {self.run_checkpoint_path}")
        except Exception:
            self.status = "failed"
            raise
        finally:
            self.metrics_logger.close()
            if self.sample_logger is not None:
                self.sample_logger.close()
            if self.visual_logger is not None:
                self.visual_logger.close()
            self._finalise_metadata(step)
            self._emit_progress(
                event="finished",
                status=self.status,
                step=step,
                run_dir=str(self.run_dir) if self.run_dir else None,
                metadata_path=str(self.metadata_path) if self.metadata_path else None,
            )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a tiny Transformer on subtitle transcripts")
    parser.add_argument(
        "--arch-preset",
        type=str,
        choices=sorted(MODEL_ARCH_PRESETS),
        default=None,
        help="Nom d'un preset d'architecture (mini-gpt, small-gpt, …)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Répertoire contenant les .txt nettoyés (défaut: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--extra-data-dir",
        dest="extra_data_dirs",
        type=Path,
        action="append",
        default=None,
        help="Répertoire additionnel à concaténer (option répétable)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Longueur de contexte (défaut: 256)",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=None,
        help="Longueur minimale aléatoire utilisée lors de l'entraînement (0 pour auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Taille de batch entraînement (défaut: 16)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Dimension des embeddings Transformer (défaut: 128)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Taille du vocabulaire/tokenizer (défaut: 256)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Nombre de têtes d'attention (défaut: 4)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Nombre de blocs Transformer (défaut: 4)",
    )
    parser.add_argument(
        "--ff-hidden-dim",
        type=int,
        default=None,
        help="Dimension cachée du MLP (défaut: 512)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Taux de dropout (défaut: 0.1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Nombre d'étapes d'entraînement (défaut: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (défaut: 3e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay pour AdamW (défaut: 0.0)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Pas entre évaluations (défaut: 50)",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Batchs utilisés pour l'éval (défaut: 10)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Fraction dédiée au train (défaut: 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine aléatoire (défaut: 13)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Périphérique, ex: 'cpu' ou 'cuda:0' (défaut: auto)",
    )
    parser.add_argument(
        "--log-dir",
        type=optional_path,
        default=argparse.SUPPRESS,
        help="Directory used to store run logs (use 'none' to disable logging)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=argparse.SUPPRESS,
        help="Optional name for the logging run directory",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=argparse.SUPPRESS,
        help="Steps interval for text sampling preview (0 to disable)",
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default=argparse.SUPPRESS,
        help="Prompt used when generating the preview text",
    )
    parser.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum tokens generated for each preview",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=argparse.SUPPRESS,
        help="Sampling temperature for previews",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=argparse.SUPPRESS,
        help="Top-k sampling cutoff for previews (0 to disable)",
    )
    parser.add_argument(
        "--metrics-log-fraction",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of total steps at which training loss is logged (e.g. 0.05 pour 5%)",
    )
    parser.add_argument(
        "--resume-from",
        dest="resume_checkpoint",
        type=Path,
        default=None,
        help="Chemin d'un checkpoint (.pt) pour reprendre l'entraînement",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        default=None,
        help="Répertoire de run existant à réutiliser pour journaux/échantillons",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.arch_preset is not None:
        apply_arch_preset(cfg, args.arch_preset)
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.extra_data_dirs:
        cfg.extra_data_dirs = list(args.extra_data_dirs)
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.min_seq_len is not None:
        cfg.min_seq_len = max(0, args.min_seq_len)
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.embed_dim is not None:
        cfg.embed_dim = args.embed_dim
    if args.vocab_size is not None:
        cfg.vocab_size = args.vocab_size
    if args.num_heads is not None:
        cfg.num_heads = args.num_heads
    if args.num_layers is not None:
        cfg.num_layers = args.num_layers
    if args.ff_hidden_dim is not None:
        cfg.ff_hidden_dim = args.ff_hidden_dim
    if args.dropout is not None:
        cfg.dropout = args.dropout
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.eval_batches is not None:
        cfg.eval_batches = args.eval_batches
    if args.train_split is not None:
        cfg.train_split = args.train_split
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if hasattr(args, "log_dir"):
        cfg.log_dir = args.log_dir
    if hasattr(args, "run_name"):
        cfg.run_name = args.run_name
    if hasattr(args, "sample_interval"):
        cfg.sample_interval = args.sample_interval
    if hasattr(args, "sample_prompt"):
        cfg.sample_prompt = args.sample_prompt
    if hasattr(args, "sample_max_new_tokens"):
        cfg.sample_max_new_tokens = args.sample_max_new_tokens
    if hasattr(args, "sample_temperature"):
        cfg.sample_temperature = args.sample_temperature
    if hasattr(args, "sample_top_k"):
        cfg.sample_top_k = args.sample_top_k
    if hasattr(args, "metrics_log_fraction"):
        cfg.metrics_log_fraction = args.metrics_log_fraction
    if args.resume_checkpoint is not None:
        cfg.resume_checkpoint = args.resume_checkpoint
    if args.resume_run_dir is not None:
        cfg.resume_run_dir = args.resume_run_dir
    return cfg


def main() -> None:
    cfg = parse_args()
    trainer = SubtitleTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
