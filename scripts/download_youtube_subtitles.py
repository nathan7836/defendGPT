#!/usr/bin/env python3
"""Fetch YouTube subtitles per channel and persist them for corpus building."""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    import yt_dlp  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency guard
    sys.exit("Le paquet 'yt-dlp' est requis. Installez-le via 'pip install yt-dlp'.")


@dataclass(frozen=True)
class ChannelSpec:
    slug: str
    url: str

    def __post_init__(self) -> None:
        if not self.slug:
            raise ValueError("slug vide pour une chaîne YouTube")
        if not self.url.startswith("http"):
            raise ValueError("url invalide pour une chaîne YouTube")


DEFAULT_CHANNELS: Tuple[ChannelSpec, ...] = (
    ChannelSpec("squeezie", "https://www.youtube.com/@Squeezie"),
    ChannelSpec("tiboinshape", "https://www.youtube.com/@TiboInShape"),
    ChannelSpec("seb", "https://www.youtube.com/@SEBFRIT"),
    ChannelSpec("lenasituations", "https://www.youtube.com/@LenaSituations"),
    ChannelSpec("michou", "https://www.youtube.com/@Michou"),
    ChannelSpec("legrandjd", "https://www.youtube.com/@legrandjd"),
    ChannelSpec("montcorvo", "https://www.youtube.com/@MontCorvo"),
    ChannelSpec("cyprien", "https://www.youtube.com/@cyprien"),
    ChannelSpec("mistervofficial", "https://www.youtube.com/@mistervofficial"),
    ChannelSpec("superkevintran", "https://www.youtube.com/@superkevintran"),
    ChannelSpec("inoxtag", "https://www.youtube.com/@inoxtag"),
    ChannelSpec("lefatshow", "https://www.youtube.com/@LeFatShow"),
    ChannelSpec("micode", "https://www.youtube.com/@Micode"),
    ChannelSpec("arte", "https://www.youtube.com/@arte"),
    ChannelSpec("baladementale", "https://www.youtube.com/@BaladeMentale"),
    ChannelSpec("hugodecrypte-grands-formats", "https://www.youtube.com/@hugodecryptegrandsformats"),
    ChannelSpec("hugodecrypteactus", "https://www.youtube.com/@hugodecrypteactus"),
    ChannelSpec("nowtech", "https://www.youtube.com/@Nowtech"),
    ChannelSpec("gotaga", "https://www.youtube.com/@Gotaga"),
    ChannelSpec("djilsi", "https://www.youtube.com/@Djilsi"),
    ChannelSpec("bazardugrenier", "https://www.youtube.com/@BazarduGrenier"),
    ChannelSpec("joueurdugrenier", "https://www.youtube.com/@joueurdugrenier"),
    ChannelSpec("wankilfrvod", "https://www.youtube.com/@WankilFrVOD"),
    ChannelSpec("ego_one", "https://www.youtube.com/@ego_one"),
    ChannelSpec("wankilfr", "https://www.youtube.com/@wankilfr"),
)

DEFAULT_LANG_CODES: Tuple[str, ...] = ("fr", "fr-FR")

LANG_SLUG = re.compile(r"[^a-z0-9_]+")


def slugify(label: str) -> str:
    lowered = label.strip().lower()
    cleaned = LANG_SLUG.sub("-", lowered)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "channel"


def normalize_language(code: str) -> str:
    cleaned = code.strip().replace("_", "-")
    if not cleaned:
        return ""
    parts = [part for part in cleaned.split("-") if part]
    if not parts:
        return ""
    normalized: List[str] = []
    for idx, part in enumerate(parts):
        normalized.append(part.lower() if idx == 0 else part.upper())
    return "-".join(normalized)


def parse_channel_arg(value: str) -> ChannelSpec:
    if "=" in value:
        slug_part, url_part = value.split("=", 1)
        slug = slugify(slug_part)
        url = url_part.strip()
    else:
        url = value.strip()
        handle = url.rstrip("/").split("/")[-1].lstrip("@")
        slug = slugify(handle)
    return ChannelSpec(slug=slug, url=url)


def read_archive(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if stripped:
            lines.append(stripped)
    return set(lines)


def download_channel(
    channel: ChannelSpec,
    output_root: Path,
    *,
    languages: Sequence[str],
    include_automatic: bool,
    playlist_end: Optional[int],
    force_refresh: bool,
    quiet: bool,
    progress_bar: Optional[tqdm] = None,
) -> Dict[str, object]:
    channel_dir = output_root / channel.slug
    subtitle_dir = channel_dir / "subtitles"
    temp_dir = channel_dir / "tmp"
    archive_path = channel_dir / "download_archive.txt"

    channel_dir.mkdir(parents=True, exist_ok=True)
    subtitle_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if force_refresh and archive_path.exists():
        archive_path.unlink()

    before = read_archive(archive_path)

    opts: Dict[str, object] = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": include_automatic,
        "subtitlesformat": "vtt",
        "outtmpl": "%(id)s.%(ext)s",
        "quiet": quiet,
        "no_warnings": quiet,
        "ignoreerrors": True,
        "download_archive": str(archive_path),
        "paths": {
            "home": str(channel_dir),
            "temp": str(temp_dir),
            "subtitle": str(subtitle_dir),
        },
    }

    if languages:
        opts["subtitleslangs"] = list(languages)
    else:
        opts["subtitleslangs"] = ["all"]
    if playlist_end is not None:
        opts["playlistend"] = playlist_end

    completed_files: Set[str] = set()

    def on_progress(status: Dict[str, object]) -> None:
        if status.get("status") != "finished":
            return
        filename = status.get("filename")
        if not isinstance(filename, str):
            return
        if filename in completed_files:
            return
        completed_files.add(filename)
        if progress_bar is not None:
            progress_bar.update(1)

    opts["progress_hooks"] = [on_progress]

    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([channel.url])
        except yt_dlp.utils.DownloadError as err:  # type: ignore[attr-defined]
            print(f"[youtube] erreur pour {channel.slug}: {err}", file=sys.stderr)

    after = read_archive(archive_path)
    new_entries = sorted(after.difference(before))

    summary = {
        "channel": channel.slug,
        "url": channel.url,
        "new_items": len(new_entries),
        "total_items": len(after),
    }
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge les sous-titres YouTube pour un ensemble de chaînes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "youtube_channels",
        help="Dossier racine recevant les sous-titres par chaîne (défaut: data/youtube_channels)",
    )
    parser.add_argument(
        "--channel",
        dest="channels",
        action="append",
        default=None,
        help="Chaîne supplémentaire au format slug=url ou directement une URL.",
    )
    parser.add_argument(
        "--only",
        dest="only",
        action="append",
        default=None,
        help="Limiter l'exécution à certains slugs de chaînes.",
    )
    parser.add_argument(
        "--language",
        dest="languages",
        action="append",
        default=None,
        help="Langue de sous-titres à récupérer (peut être répétée). Défaut: fr, fr-FR.",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Récupère toutes les langues de sous-titres disponibles.",
    )
    parser.add_argument(
        "--no-automatic",
        action="store_true",
        help="Ignore les sous-titres générés automatiquement.",
    )
    parser.add_argument(
        "--playlist-end",
        type=int,
        default=None,
        help="Limite le nombre de vidéos parcourues par chaîne (debug).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore l'archive de téléchargement et retente toutes les vidéos.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Sortie plus compacte de yt-dlp.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Nombre de chaînes traitées en parallèle (défaut: 4).",
    )
    return parser.parse_args(argv)


def resolve_channels(args: argparse.Namespace) -> List[ChannelSpec]:
    channels: List[ChannelSpec] = list(DEFAULT_CHANNELS)
    if args.channels:
        for raw in args.channels:
            channels.append(parse_channel_arg(raw))
    if args.only:
        wanted = {slugify(entry) for entry in args.only}
        channels = [ch for ch in channels if ch.slug in wanted]
    dedup: Dict[str, ChannelSpec] = {}
    for ch in channels:
        dedup[ch.slug] = ch
    return list(dedup.values())


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    channels = resolve_channels(args)
    if not channels:
        print("Aucune chaîne à traiter.")
        return 0

    languages: Sequence[str]
    if args.all_languages:
        languages = []
    elif args.languages:
        cleaned = [normalize_language(lang) for lang in args.languages if lang.strip()]
        languages = tuple(lang for lang in cleaned if lang)
        if not languages:
            languages = tuple(normalize_language(code) for code in DEFAULT_LANG_CODES)
    else:
        languages = tuple(normalize_language(code) for code in DEFAULT_LANG_CODES)

    worker_count = max(1, args.workers)
    worker_count = min(worker_count, len(channels))

    overall_bar: Optional[tqdm]
    channel_bars: Dict[str, Optional[tqdm]]
    if args.quiet:
        overall_bar = None
        channel_bars = {}
    else:
        overall_bar = tqdm(total=len(channels), desc="Chaînes", unit="chaîne", position=0)
        channel_bars = {
            channel.slug: tqdm(
                desc=f"{channel.slug}",
                unit="sous-titre",
                leave=False,
                position=idx,
            )
            for idx, channel in enumerate(channels, start=1)
        }

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_channel = {
                executor.submit(
                    download_channel,
                    channel,
                    args.output_root,
                    languages=languages,
                    include_automatic=not args.no_automatic,
                    playlist_end=args.playlist_end,
                    force_refresh=args.force_refresh,
                    quiet=args.quiet,
                    progress_bar=channel_bars.get(channel.slug),
                ): channel
                for channel in channels
            }

            for future in as_completed(future_to_channel):
                channel = future_to_channel[future]
                bar = channel_bars.get(channel.slug)
                try:
                    summary = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    message = f"[youtube] erreur inattendue pour {channel.slug}: {exc}"
                    if overall_bar is not None:
                        overall_bar.update(1)
                        overall_bar.write(message)
                    else:
                        print(message, file=sys.stderr)
                    if bar is not None:
                        bar.set_description(f"{channel.slug} (erreur)")
                        bar.close()
                        channel_bars[channel.slug] = None
                    continue

                if overall_bar is not None:
                    overall_bar.update(1)
                    overall_bar.write(
                        f"[youtube] {channel.slug}: +{summary['new_items']} nouveaux sous-titres (total {summary['total_items']})"
                    )
                else:
                    print(
                        f"[youtube] {channel.slug}: +{summary['new_items']} nouveaux sous-titres (total {summary['total_items']})"
                    )

                if bar is not None:
                    bar.set_postfix(new=summary["new_items"], total=summary["total_items"])
                    bar.close()
                    channel_bars[channel.slug] = None
    finally:
        if overall_bar is not None:
            overall_bar.close()
        for bar in channel_bars.values():
            if bar is not None:
                bar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
