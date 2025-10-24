#!/usr/bin/env python3
"""Fetch and store a large corpus of Wikipedia articles related to AI."""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_CLEAN_DIR = ROOT / "data_clean" / "wikipedia"

# Default category seeds per language (falls back to English seeds).
DEFAULT_CATEGORY_SEEDS = {
    "fr": [
        "Catégorie:Intelligence artificielle",
        "Catégorie:Apprentissage automatique",
        "Catégorie:Apprentissage profond",
        "Catégorie:Robotique",
        "Catégorie:Traitement automatique du langage naturel",
        "Catégorie:Informatique",
        "Catégorie:Informatique théorique",
        "Catégorie:Science des données",
        "Catégorie:Statistiques",
        "Catégorie:Technologies de l'information",
        "Catégorie:Technologies de l'information et de la communication",
    ],
    "en": [
        "Category:Artificial intelligence",
        "Category:Machine learning",
        "Category:Deep learning",
        "Category:Robotics",
        "Category:Natural language processing",
        "Category:Computer science",
        "Category:Data science",
        "Category:Statistics",
        "Category:Information technology",
        "Category:Information and communications technology",
    ],
}

# Wikipedia only serves full-article extracts one at a time, so we fetch pages individually.
EXTRACT_BATCH_SIZE = 1


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "article"


HEADERS = {
    "User-Agent": "DefendGPT-DataPrep/1.1 (+https://github.com/anisayari/defendGPT)"
}


def chunked(iterable: Iterable[str], size: int) -> Iterator[Tuple[str, ...]]:
    iterator = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def iter_category_members(
    session: requests.Session,
    language: str,
    category: str,
    cmtype: str = "page|subcat",
) -> Iterator[dict]:
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "format": "json",
        "cmtitle": category,
        "cmlimit": "max",
        "cmtype": cmtype,
        "cmprop": "ids|title|type|timestamp",
    }
    while True:
        response = session.get(endpoint, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        payload = response.json()
        members = payload.get("query", {}).get("categorymembers", [])
        for member in members:
            yield member
        cont = payload.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
        # Tiny pause to play nice with the API if a category is huge.
        time.sleep(0.1)


def gather_ai_titles(
    session: requests.Session,
    language: str,
    min_articles: int,
    category_seeds: Iterable[str],
    max_depth: int,
    *,
    progress_every: int = 0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Breadth-first crawl of AI categories returning page titles → pageid."""

    pages: Dict[str, int] = {}
    categories_seen: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque()

    for seed in category_seeds:
        normalized = seed.strip()
        if not normalized:
            continue
        queue.append((normalized, 0))

    next_page_report = progress_every if progress_every else 0
    next_category_report = progress_every if progress_every else 0

    while queue and len(pages) < min_articles:
        category, depth = queue.popleft()
        if category in categories_seen:
            continue
        categories_seen[category] = depth
        if progress_every and len(categories_seen) == 1:
            if progress_cb:
                progress_cb(
                    f"[{language}] Processing category '{category}' (depth {depth})"
                )
        elif progress_every and next_category_report and len(categories_seen) >= next_category_report:
            if progress_cb:
                progress_cb(
                    f"[{language}] Visited {len(categories_seen)} categories; discovered {len(pages)} articles so far"
                )
            next_category_report += progress_every
        for member in iter_category_members(session, language, category):
            title = member.get("title")
            if not title:
                continue
            ns = member.get("ns")
            page_id = member.get("pageid")
            if ns == 0 and page_id:
                pages.setdefault(title, page_id)
                if progress_every and len(pages) == 1:
                    if progress_cb:
                        progress_cb(
                            f"[{language}] Collected first title '{title}'"
                        )
                elif progress_every and next_page_report and len(pages) >= next_page_report:
                    if progress_cb:
                        progress_cb(
                            f"[{language}] Collected {len(pages)} titles towards target {min_articles}"
                        )
                    next_page_report += progress_every
                if len(pages) >= min_articles:
                    break
            elif ns == 14 and depth < max_depth:
                queue.append((title, depth + 1))
        # Small pause between categories to avoid hammering the API.
        time.sleep(0.05)

    return pages, categories_seen


def fetch_extracts(
    session: requests.Session,
    language: str,
    titles: Iterable[str],
) -> Iterator[Tuple[str, int, str]]:
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    for batch in chunked(titles, EXTRACT_BATCH_SIZE):
        params = {
            "action": "query",
            "prop": "extracts",
            "format": "json",
            "redirects": 1,
            "explaintext": 1,
            "exlimit": str(len(batch)),
            "titles": "|".join(batch),
        }
        response = session.get(endpoint, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract")
            if not extract:
                continue
            title = page.get("title")
            page_id = page.get("pageid")
            if not title or page_id is None:
                continue
            header = f"# {title}\n"
            yield title, int(page_id), header + extract.strip() + "\n"
        # Rate limit ever so slightly between batches.
        time.sleep(0.05)


def normalize_topics(topics: Iterable[str] | str | None) -> List[str]:
    if topics is None:
        return []
    if isinstance(topics, str):
        # Allow comma-separated values from API payloads
        return [item.strip() for item in topics.split(",") if item.strip()]
    return [str(item).strip() for item in topics if str(item).strip()]


def prepare_wikipedia(
    topics: Iterable[str] | str | None = None,
    language: str = "fr",
    *,
    min_articles: int = 10_000,
    category_seeds: Optional[Iterable[str]] = None,
    max_category_depth: int = 3,
    progress_every: int = 500,
) -> Dict[str, int | str]:
    """Fetch Wikipedia content focused on AI until ``min_articles`` are stored."""

    start_time = time.time()
    normalized_topics = normalize_topics(topics)
    DATA_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Track counts and metadata for the return payload.
    saved = 0
    total_bytes = 0
    skipped_missing = 0
    slug_collisions = 0
    slug_counts: Dict[str, int] = {}
    final_slugs: set[str] = set()
    titles: Dict[str, Optional[int]] = {}
    crawled_categories: Dict[str, int] = {}

    def report(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    progress_every = max(progress_every, 0)

    with requests.Session() as session:
        if normalized_topics:
            titles = {topic: None for topic in normalized_topics}
        else:
            seeds = list(category_seeds or DEFAULT_CATEGORY_SEEDS.get(language, []))
            if not seeds:
                seeds = list(DEFAULT_CATEGORY_SEEDS["en"])
            if progress_every:
                report(
                    f"[{language}] Starting category crawl with {len(seeds)} seed(s); target {min_articles} articles"
                )
            titles_map, crawled_categories = gather_ai_titles(
                session,
                language,
                min_articles,
                seeds,
                max_category_depth,
                progress_every=progress_every,
                progress_cb=report if progress_every else None,
            )
            titles = {title: page_id for title, page_id in titles_map.items()}
            if progress_every:
                report(
                    f"[{language}] Finished crawl -> {len(titles)} titles discovered across {len(crawled_categories)} categories"
                )

        if not titles:
            raise RuntimeError("No topics available to download from Wikipedia")

        if progress_every:
            report(f"[{language}] Starting article downloads for {len(titles)} titles")

        for title, page_id, content in fetch_extracts(session, language, titles.keys()):
            base_slug = slugify(title)
            count = slug_counts.get(base_slug, 0)
            slug = base_slug if count == 0 else f"{base_slug}_{page_id}"
            while slug in final_slugs:
                count += 1
                slug = f"{base_slug}_{page_id}_{count}"
            if count > 0:
                slug_collisions += 1
            slug_counts[base_slug] = count + 1
            final_slugs.add(slug)

            path = DATA_CLEAN_DIR / f"{slug}.txt"
            path.write_text(content, encoding="utf-8")
            saved += 1
            total_bytes += path.stat().st_size
            if progress_every and (saved == 1 or saved % progress_every == 0):
                report(
                    f"[{language}] Saved {saved} article files (~{total_bytes // 1024} KiB accumulated)"
                )
            if saved >= min_articles and not normalized_topics:
                break

    if progress_every:
        report(f"[{language}] Completed run -> {saved} articles saved")

    expected = len(titles) if normalized_topics else min_articles
    if saved < expected:
        skipped_missing = expected - saved

    summary: Dict[str, int | str] = {
        "articles": saved,
        "size_bytes": total_bytes,
        "skipped": skipped_missing,
        "slug_collisions": slug_collisions,
        "target": min_articles if not normalized_topics else len(titles),
        "language": language,
        "elapsed_seconds": int(round(time.time() - start_time)),
    }

    if normalized_topics:
        summary["requested_topics"] = len(titles)
    else:
        summary["discovered_titles"] = len(titles)
        summary["categories_visited"] = len(crawled_categories)
        summary["max_depth_reached"] = max(crawled_categories.values(), default=0)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Wikipedia articles about AI topics")
    parser.add_argument("--language", default="fr", help="Wikipedia language code (default: fr)")
    parser.add_argument("--min-articles", type=int, default=10_000, help="Minimum number of articles to store")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum category depth when crawling")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Report progress after this many items (set to 0 to disable)",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="Optional category seed (can be set multiple times)",
    )
    parser.add_argument("topics", nargs="*", help="Explicit list of article titles to fetch")
    args = parser.parse_args()

    category_seeds = args.categories
    summary = prepare_wikipedia(
        args.topics or None,
        language=args.language,
        min_articles=args.min_articles,
        category_seeds=category_seeds,
        max_category_depth=args.max_depth,
        progress_every=args.progress_every,
    )
    print("Wikipedia data prepared -> " + json.dumps(summary))

if __name__ == "__main__":
    main()
