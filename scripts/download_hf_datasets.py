#!/usr/bin/env python3
"""Download and persist a curated set of Hugging Face datasets."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover - guardrail
    sys.exit(
        "Le paquet 'datasets' est requis. Installez-le via 'pip install datasets'."
    )


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config: Optional[str] = None
    split: Optional[str] = None

    def slug(self) -> str:
        pieces = [self.name]
        if self.config:
            pieces.append(self.config)
        if self.split:
            pieces.append(self.split)
        raw = "__".join(pieces)
        slug = raw.replace("/", "__").replace(" ", "_")
        slug = slug.replace("-", "_")
        return slug.lower()

    def label(self) -> str:
        label = self.name
        if self.config:
            label += f" ({self.config})"
        if self.split:
            label += f" [{self.split}]"
        return label


DEFAULT_DATASETS: Sequence[DatasetSpec] = (
    DatasetSpec("PleIAs/French-PD-Books"),
    DatasetSpec("Kant1/French_Wikipedia_articles"),
    DatasetSpec("almanach/hc3_french_ood", "faq_fr_gouv"),
    DatasetSpec("almanach/hc3_french_ood", "faq_fr_random"),
    DatasetSpec("almanach/hc3_french_ood", "hc3_en_full"),
    DatasetSpec("manu/old_french_30b_separate"),
)


def _deduplicate(specs: Iterable[DatasetSpec]) -> List[DatasetSpec]:
    seen = set()
    ordered: List[DatasetSpec] = []
    for spec in specs:
        key = (spec.name, spec.config, spec.split)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(spec)
    return ordered


def _save_dataset(dataset: Dataset | DatasetDict, target_dir: Path) -> str:
    dataset.save_to_disk(target_dir)
    if isinstance(dataset, DatasetDict):
        summary = ", ".join(
            f"{split}:{ds.num_rows}" for split, ds in dataset.items()
        )
    else:
        summary = f"rows={dataset.num_rows}"
    return summary


def download_datasets(
    specs: Sequence[DatasetSpec],
    *,
    output_root: Path,
    cache_dir: Optional[Path],
    force: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for spec in _deduplicate(specs):
        target_dir = output_root / spec.slug()
        print(f"[datasets] {spec.label()}")

        if target_dir.exists():
            if not force:
                print(f"  -> ignoré (présent): {target_dir}")
                continue
            print(f"  -> suppression du dossier existant: {target_dir}")
            shutil.rmtree(target_dir)

        load_kwargs = {}
        if spec.config:
            load_kwargs["name"] = spec.config
        if spec.split:
            load_kwargs["split"] = spec.split
        if cache_dir:
            load_kwargs["cache_dir"] = str(cache_dir)

        try:
            dataset = load_dataset(spec.name, **load_kwargs)
        except Exception as exc:  # pragma: no cover - network issues
            print(f"  !! téléchargement échoué: {exc}")
            continue

        try:
            summary = _save_dataset(dataset, target_dir)
        except Exception as exc:  # pragma: no cover - disk issues
            print(f"  !! sauvegarde échouée: {exc}")
            continue

        print(f"  -> sauvegardé dans {target_dir} ({summary})")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge un ensemble de jeux de données Hugging Face et les stocke sur disque.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "hf_datasets",
        help="Dossier racine pour les données téléchargées (défaut: data/hf_datasets)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data") / "hf_cache",
        help="Cache Hugging Face (défaut: data/hf_cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Réécrit les dossiers existants au lieu de les ignorer",
    )
    parser.add_argument(
        "--no-cache-dir",
        action="store_true",
        help="N'utilise pas de cache local pour load_dataset",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cache_dir = None if args.no_cache_dir else args.cache_dir

    download_datasets(
        DEFAULT_DATASETS,
        output_root=args.output_root,
        cache_dir=cache_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
