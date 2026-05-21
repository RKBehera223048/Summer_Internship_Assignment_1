"""
Task 9: Context-Aware Scraping Pipeline

Extends the Task 8 scraper by:
    1. Reading keywords from ``keywords.txt``
    2. Searching for each keyword using :class:`TwitterScraper`
    3. Merging all results and removing duplicates (by ``tweet_link``)
    4. Categorising tweets and organising media into context folders
    5. Generating per-category metadata JSON files
    6. Saving merged results as CSV and JSON

Category mapping
----------------
- **ai**            : AI, artificial intelligence, machine learning, deep learning, AI generated
- **robotics**      : robot, robotics, automation
- **sports**        : sport, game, highlight, football, cricket
- **semiconductor** : semiconductor, chip, silicon, wafer, self-driving, autonomous

Author : Assignment 1 -- IIT Bhubaneswar Summer Internship
Date   : 2026-05-21
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

# Import the scraper from Task 8
from task8_web_scraper import TwitterScraper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base path – relative to this script's location
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Category mapping (keyword fragments -> category label)
# ---------------------------------------------------------------------------
CATEGORY_MAP: dict[str, list[str]] = {
    "ai": [
        "ai",
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "ai generated",
    ],
    "robotics": [
        "robot",
        "robotics",
        "automation",
    ],
    "sports": [
        "sport",
        "game",
        "highlight",
        "football",
        "cricket",
    ],
    "semiconductor": [
        "semiconductor",
        "chip",
        "silicon",
        "wafer",
        "self-driving",
        "autonomous",
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_keywords(filepath: Path) -> list[str]:
    """Read keywords from a text file (one keyword per line).

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to the keywords file.

    Returns
    -------
    list[str]
        Stripped, non-empty lines from the file.
    """
    if not filepath.exists():
        logger.error("Keywords file not found: %s", filepath)
        return []

    lines = filepath.read_text(encoding="utf-8").splitlines()
    keywords = [line.strip() for line in lines if line.strip()]
    logger.info("[FILE] Loaded %d keywords from %s", len(keywords), filepath)
    return keywords


def categorise_tweet(tweet: dict) -> str:
    """Determine the category of a tweet based on its keyword and text.

    The function checks the tweet's ``keyword``, ``context``, and
    ``tweet_text`` fields against the :data:`CATEGORY_MAP`. The first
    matching category wins; if nothing matches the tweet is labelled
    ``'other'``.

    Parameters
    ----------
    tweet : dict
        A tweet dictionary produced by :class:`TwitterScraper`.

    Returns
    -------
    str
        Category label (e.g. ``'ai'``, ``'robotics'``, ``'sports'``,
        ``'semiconductor'``, or ``'other'``).
    """
    searchable = " ".join(
        [
            tweet.get("keyword", ""),
            tweet.get("context", ""),
            tweet.get("tweet_text", ""),
        ]
    ).lower()

    for category, fragments in CATEGORY_MAP.items():
        for fragment in fragments:
            if fragment in searchable:
                return category

    return "other"


def remove_duplicates(tweets: list[dict]) -> tuple[list[dict], int]:
    """De-duplicate tweets by ``tweet_link``.

    Parameters
    ----------
    tweets : list[dict]
        Full list of tweets (may contain duplicates).

    Returns
    -------
    tuple[list[dict], int]
        A tuple of (unique_tweets, count_of_removed_duplicates).
    """
    seen: set[str] = set()
    unique: list[dict] = []

    for tweet in tweets:
        link = tweet.get("tweet_link", "")
        if link and link in seen:
            continue
        seen.add(link)
        unique.append(tweet)

    removed = len(tweets) - len(unique)
    return unique, removed


def organise_by_category(
    tweets: list[dict],
    output_base: Path,
) -> dict[str, list[dict]]:
    """Sort tweets into category buckets and create folder structures.

    For each category a directory is created under
    ``<output_base>/videos/<category>/`` and a ``metadata.json`` is written
    listing the associated video URLs.

    Parameters
    ----------
    tweets : list[dict]
        De-duplicated tweet list.
    output_base : Path
        Root output directory (e.g. ``scraper_output/``).

    Returns
    -------
    dict[str, list[dict]]
        Mapping of category -> list of tweet dicts belonging to it.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)

    for tweet in tweets:
        cat = categorise_tweet(tweet)
        tweet["category"] = cat
        buckets[cat].append(tweet)

    # Create category directories and metadata files
    for cat, cat_tweets in buckets.items():
        cat_dir = output_base / "videos" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "category": cat,
            "total_videos": len(cat_tweets),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "videos": [
                {
                    "username": t["username"],
                    "tweet_link": t["tweet_link"],
                    "video_url": t["video_url"],
                    "tweet_text": t["tweet_text"][:120],
                    "likes": t["likes"],
                    "reposts": t["reposts"],
                    "timestamp": t["timestamp"],
                }
                for t in cat_tweets
            ],
        }

        meta_path = cat_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[DIR] %s/ -> %d tweets  (metadata saved)", cat, len(cat_tweets))

    return dict(buckets)


def print_summary(
    total_collected: int,
    duplicates_removed: int,
    category_counts: dict[str, int],
) -> None:
    """Print a human-readable summary of the pipeline run.

    Parameters
    ----------
    total_collected : int
        Number of tweets before de-duplication.
    duplicates_removed : int
        Number of duplicate tweets removed.
    category_counts : dict[str, int]
        Mapping of category -> tweet count.
    """
    print("\n" + "=" * 70)
    print("  Task 9 -- Context Scraping Pipeline -- SUMMARY")
    print("=" * 70)
    print(f"  Total tweets collected (raw)  : {total_collected}")
    print(f"  Duplicates removed            : {duplicates_removed}")
    print(f"  Unique tweets retained        : {total_collected - duplicates_removed}")
    print()
    print("  Per-category breakdown:")
    for cat, count in sorted(category_counts.items()):
        print(f"    * {cat:<16s} : {count}")
    print("=" * 70 + "\n")


# ======================================================================
# Main execution
# ======================================================================


def main() -> None:
    """Run the context-aware scraping pipeline."""
    print("\n" + "=" * 70)
    print("  Task 9 -- Context-Aware Scraping Pipeline")
    print("=" * 70 + "\n")

    # --- Step 1: Load keywords ----------------------------------------
    keywords_path = BASE_DIR / "keywords.txt"
    keywords = load_keywords(keywords_path)

    if not keywords:
        print("[ERROR] No keywords found. Please create keywords.txt first.")
        return

    print(f"  Keywords loaded: {keywords}\n")

    # --- Step 2: Scrape for each keyword ------------------------------
    scraper = TwitterScraper(max_retries=3, retry_delay=2)
    all_tweets: list[dict] = []

    for kw in keywords:
        # Derive a short context from the keyword itself
        context = kw.strip().title()
        tweets = scraper.search_tweets(kw, context, max_results=20)
        all_tweets.extend(tweets)
        print(f"  [OK] Collected {len(tweets)} tweets for '{kw}'")

    total_collected = len(all_tweets)
    print(f"\n  Total raw tweets: {total_collected}\n")

    # --- Step 3: Remove duplicates ------------------------------------
    unique_tweets, dups_removed = remove_duplicates(all_tweets)
    print(f"  Duplicates removed: {dups_removed}")
    print(f"  Unique tweets     : {len(unique_tweets)}\n")

    # --- Step 4: Categorise & organise into folders -------------------
    output_dir = BASE_DIR / "scraper_output"
    buckets = organise_by_category(unique_tweets, output_dir)

    category_counts = {cat: len(items) for cat, items in buckets.items()}

    # --- Step 5: Save merged results ----------------------------------
    scraper.save_to_csv(unique_tweets, output_dir / "merged_tweets.csv")
    scraper.save_to_json(unique_tweets, output_dir / "merged_tweets.json")

    # --- Step 6: Print summary ----------------------------------------
    print_summary(total_collected, dups_removed, category_counts)

    print(f"  Merged CSV  : {output_dir / 'merged_tweets.csv'}")
    print(f"  Merged JSON : {output_dir / 'merged_tweets.json'}")
    print(f"  Video dirs  : {output_dir / 'videos' / '<category>' }\n")


if __name__ == "__main__":
    main()
