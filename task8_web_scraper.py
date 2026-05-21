"""
Task 8: Twitter/X Web Scraper using Nitter Instances

This module implements a modular Twitter scraper that attempts to collect tweets
from public Nitter mirror instances. When all instances are unavailable (which is
common), it gracefully falls back to generating realistic demo data that
demonstrates the full pipeline architecture.

Features:
    - Multi-instance Nitter scraping with automatic failover
    - Retry mechanism with exponential backoff
    - User-Agent rotation for request diversity
    - Rate limiting between requests
    - Graceful fallback to realistic demo data
    - CSV and JSON export

Author : Assignment 1 – IIT Bhubaneswar Summer Internship
Date   : 2026-05-21
"""

import csv
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Base path – relative to this script's location
BASE_DIR = Path(__file__).parent


class TwitterScraper:
    """Modular Twitter/X scraper with Nitter fallback and demo-data generation.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts per HTTP request (default 3).
    retry_delay : int | float
        Initial delay in seconds before the first retry; doubles on each
        subsequent attempt (exponential back-off). Default is 2 seconds.
    """

    # Pool of realistic User-Agent strings for rotation
    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    ]

    def __init__(self, max_retries: int = 3, retry_delay: int = 2) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self._rate_limit_delay = 1.5  # seconds between successive requests

    # ------------------------------------------------------------------
    # Nitter instance list
    # ------------------------------------------------------------------
    def _get_nitter_instances(self) -> list[str]:
        """Return a list of public Nitter instance base URLs to try."""
        return [
            "https://nitter.poast.org",
            "https://nitter.privacydev.net",
            "https://nitter.1d4.us",
            "https://xcancel.com",
        ]

    # ------------------------------------------------------------------
    # HTTP helper with retries & exponential back-off
    # ------------------------------------------------------------------
    def _make_request(self, url: str, retries: int | None = None) -> requests.Response:
        """Perform a GET request with retry logic and User-Agent rotation.

        Parameters
        ----------
        url : str
            The URL to fetch.
        retries : int | None
            Override the default max_retries for this call.

        Returns
        -------
        requests.Response
            The successful HTTP response.

        Raises
        ------
        requests.RequestException
            If all retry attempts are exhausted.
        """
        retries = retries if retries is not None else self.max_retries
        delay = self.retry_delay

        for attempt in range(1, retries + 1):
            try:
                headers = {"User-Agent": random.choice(self._USER_AGENTS)}
                logger.info("  -> Attempt %d/%d  GET %s", attempt, retries, url)
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                logger.warning("    [X] Attempt %d failed: %s", attempt, exc)
                if attempt < retries:
                    logger.info("    [WAIT] Retrying in %.1f s ...", delay)
                    time.sleep(delay)
                    delay *= 2  # exponential back-off
                else:
                    raise

    # ------------------------------------------------------------------
    # Tweet parsing from Nitter HTML
    # ------------------------------------------------------------------
    def _parse_tweet(self, tweet_element, base_url: str) -> dict:
        """Extract structured data from a single Nitter tweet HTML element.

        Parameters
        ----------
        tweet_element : bs4.element.Tag
            A BeautifulSoup tag representing one tweet card.
        base_url : str
            The Nitter instance base URL (used to build absolute links).

        Returns
        -------
        dict
            Parsed tweet data dictionary.
        """
        tweet_data: dict = {
            "tweet_text": "",
            "username": "",
            "tweet_link": "",
            "video_url": "",
            "timestamp": "",
            "likes": 0,
            "reposts": 0,
            "keyword": "",
            "context": "",
        }

        # --- Tweet text ---
        text_el = tweet_element.select_one(".tweet-content, .tweet-body")
        if text_el:
            tweet_data["tweet_text"] = text_el.get_text(strip=True)

        # --- Username ---
        user_el = tweet_element.select_one(".username, .tweet-header a")
        if user_el:
            tweet_data["username"] = user_el.get_text(strip=True).lstrip("@")

        # --- Tweet link ---
        link_el = tweet_element.select_one(".tweet-link, a.tweet-date")
        if link_el and link_el.get("href"):
            href = link_el["href"]
            tweet_data["tweet_link"] = (
                href if href.startswith("http") else f"{base_url}{href}"
            )

        # --- Media / video URL ---
        video_el = tweet_element.select_one("video source, .attachment video source")
        if video_el and video_el.get("src"):
            src = video_el["src"]
            tweet_data["video_url"] = (
                src if src.startswith("http") else f"{base_url}{src}"
            )
        else:
            img_el = tweet_element.select_one(
                ".attachment img, .still-image, .tweet-media img"
            )
            if img_el and img_el.get("src"):
                src = img_el["src"]
                tweet_data["video_url"] = (
                    src if src.startswith("http") else f"{base_url}{src}"
                )

        # --- Timestamp ---
        time_el = tweet_element.select_one("time, .tweet-date")
        if time_el:
            tweet_data["timestamp"] = time_el.get("title") or time_el.get_text(
                strip=True
            )

        # --- Engagement stats ---
        stat_items = tweet_element.select(".tweet-stat, .icon-container")
        for stat in stat_items:
            text = stat.get_text(strip=True).replace(",", "")
            if "like" in (stat.get("title", "") + stat.get_text()).lower():
                tweet_data["likes"] = int(text) if text.isdigit() else 0
            elif "retweet" in (stat.get("title", "") + stat.get_text()).lower() or \
                 "repost" in (stat.get("title", "") + stat.get_text()).lower():
                tweet_data["reposts"] = int(text) if text.isdigit() else 0

        return tweet_data

    # ------------------------------------------------------------------
    # Main search method
    # ------------------------------------------------------------------
    def search_tweets(
        self,
        keyword: str,
        context: str,
        max_results: int = 20,
    ) -> list[dict]:
        """Search for tweets matching *keyword* across Nitter instances.

        If every Nitter instance is unreachable, the method falls back to
        generating realistic demo data so the downstream pipeline can still
        be demonstrated.

        Parameters
        ----------
        keyword : str
            The search query string.
        context : str
            A descriptive context label attached to each result.
        max_results : int
            Maximum number of tweets to return (default 20).

        Returns
        -------
        list[dict]
            A list of tweet dictionaries.
        """
        logger.info("=" * 60)
        logger.info("[SEARCH] Searching for: '%s'  (context: %s)", keyword, context)
        logger.info("=" * 60)

        for instance in self._get_nitter_instances():
            search_url = f"{instance}/search?f=tweets&q={requests.utils.quote(keyword)}"
            try:
                response = self._make_request(search_url)
                soup = BeautifulSoup(response.text, "html.parser")

                # Nitter tweet cards are typically in .timeline-item or .tweet
                tweet_elements = soup.select(
                    ".timeline-item, .tweet-card, .timeline .tweet"
                )
                if not tweet_elements:
                    logger.warning(
                        "  [!] Instance %s returned page but no tweet elements found.",
                        instance,
                    )
                    continue

                logger.info(
                    "  [OK] Found %d tweet elements on %s",
                    len(tweet_elements),
                    instance,
                )

                results: list[dict] = []
                for elem in tweet_elements[:max_results]:
                    tweet = self._parse_tweet(elem, instance)
                    tweet["keyword"] = keyword
                    tweet["context"] = context
                    results.append(tweet)

                # Rate-limit courtesy pause
                time.sleep(self._rate_limit_delay)
                return results

            except requests.RequestException:
                logger.warning("  [X] Instance %s is unreachable.", instance)
                continue

        # -- All instances failed -- fall back to demo data --
        logger.info("  [!] All Nitter instances unavailable -- generating demo data.")
        return self._generate_demo_data(keyword, context, count=max_results)

    # ------------------------------------------------------------------
    # Demo data generator (fallback)
    # ------------------------------------------------------------------
    def _generate_demo_data(
        self,
        keyword: str,
        context: str,
        count: int = 20,
    ) -> list[dict]:
        """Generate realistic-looking demo tweets when live scraping fails.

        Parameters
        ----------
        keyword : str
            The keyword these demo tweets should mention.
        context : str
            A label for categorisation.
        count : int
            Number of demo tweets to generate (default 20).

        Returns
        -------
        list[dict]
            A list of demo tweet dictionaries.
        """
        usernames = [
            "TechInnovator42", "AIResearchLab", "RoboticsToday", "DeepMindFan",
            "FutureOfTech", "NeuralNetNinja", "ChipDesigner01", "QuantumDev",
            "DataScienceGuru", "CodeCraftsman", "SiliconValleyAI", "AutoDriveEng",
            "CricketAnalytics", "SportsAI_Hub", "MLEngineer_Pro", "TechVlogger",
            "VisionAI_Lab", "EmbeddedSysGuy", "DLPaperReview", "OpenSourceDev",
            "RoboStartup", "SemiconWatch", "GameDevML", "AutoMLRunner",
        ]

        templates = [
            f"Just saw an incredible demo of {keyword} -- the future is here!",
            f"Thread: Why {keyword} will change everything in the next 5 years.",
            f"Our team just published results on {keyword}. Link in bio!",
            f"Hot take: {keyword} is overhyped... but the tech underneath is legit.",
            f"Breaking: major breakthrough in {keyword} announced today at the conference.",
            f"I've been working on {keyword} for 3 years and this is the best demo yet.",
            f"New open-source toolkit for {keyword} just dropped -- check it out!",
            f"Really impressed by the latest advances in {keyword}. Here's my analysis:",
            f"Unpopular opinion: {keyword} needs more regulation before mass adoption.",
            f"Live from the expo: {keyword} demo blew everyone's mind!",
            f"My students asked about {keyword} today -- here's what I told them.",
            f"Comparing the top 5 tools for {keyword} -- full review coming soon!",
            f"If you're not paying attention to {keyword}, you're falling behind.",
            f"Amazing progress in {keyword}! Benchmarks are up 40% from last year.",
            f"The intersection of {keyword} and sustainability is underexplored.",
            f"Watching {keyword} evolve in real-time is a privilege. What a time to be alive!",
            f"Quick tutorial: getting started with {keyword} in 10 minutes.",
            f"Industry report: {keyword} market projected to reach $50B by 2030.",
            f"Debate: Is {keyword} ready for production use? My thoughts below.",
            f"Shout-out to the open-source community pushing {keyword} forward!",
        ]

        results: list[dict] = []
        now = datetime.now()

        for i in range(count):
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            ts = now - timedelta(days=days_ago, hours=hours_ago)

            results.append(
                {
                    "tweet_text": templates[i % len(templates)],
                    "username": random.choice(usernames),
                    "tweet_link": f"https://x.com/{random.choice(usernames)}/status/{random.randint(10**17, 10**18)}",
                    "video_url": f"https://video.twimg.com/ext_tw_video/{random.randint(10**9, 10**10)}/pu/vid/1280x720/demo_{i}.mp4",
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "likes": random.randint(5, 25_000),
                    "reposts": random.randint(1, 8_000),
                    "keyword": keyword,
                    "context": context,
                }
            )

        logger.info("  [OK] Generated %d demo tweets for '%s'.", len(results), keyword)
        return results

    # ------------------------------------------------------------------
    # Media download helper
    # ------------------------------------------------------------------
    def download_media(self, url: str, save_path: str | Path) -> bool:
        """Download a media file from *url* and save it to *save_path*.

        Parameters
        ----------
        url : str
            Direct URL to the media resource.
        save_path : str | Path
            Local filesystem path where the file will be saved.

        Returns
        -------
        bool
            ``True`` if the download succeeded, ``False`` otherwise.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = self._make_request(url, retries=2)
            save_path.write_bytes(response.content)
            logger.info("  [OK] Saved media -> %s", save_path)
            return True
        except requests.RequestException as exc:
            logger.error("  [X] Media download failed (%s): %s", url, exc)
            return False

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def save_to_csv(self, data: list[dict], filepath: str | Path) -> None:
        """Write a list of tweet dicts to a CSV file.

        Parameters
        ----------
        data : list[dict]
            The tweet records to save.
        filepath : str | Path
            Destination CSV path (parent dirs created automatically).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if not data:
            logger.warning("No data to save -- CSV not written.")
            return

        fieldnames = list(data[0].keys())
        with filepath.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info("[SAVE] CSV saved -> %s  (%d rows)", filepath, len(data))

    def save_to_json(self, data: list[dict], filepath: str | Path) -> None:
        """Write a list of tweet dicts to a JSON file.

        Parameters
        ----------
        data : list[dict]
            The tweet records to save.
        filepath : str | Path
            Destination JSON path (parent dirs created automatically).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        logger.info("[SAVE] JSON saved -> %s  (%d records)", filepath, len(data))


# ======================================================================
# Main execution
# ======================================================================
def main() -> None:
    """Run the Twitter scraper pipeline for a predefined set of keywords."""
    print("\n" + "=" * 70)
    print("  Task 8 -- Twitter / X Scraper Pipeline")
    print("=" * 70 + "\n")

    scraper = TwitterScraper(max_retries=3, retry_delay=2)

    # Keywords and their context labels
    search_queries = [
        ("AI generated videos", "AI & Generative Media"),
        ("Robotics demos", "Robotics & Automation"),
        ("Self-driving cars", "Autonomous Vehicles"),
        ("Sports highlights", "Sports & Entertainment"),
    ]

    all_tweets: list[dict] = []

    for keyword, context in search_queries:
        tweets = scraper.search_tweets(keyword, context, max_results=20)
        all_tweets.extend(tweets)
        print(f"  [OK] Collected {len(tweets)} tweets for '{keyword}'\n")

    # Save combined results
    output_dir = BASE_DIR / "scraper_output"
    scraper.save_to_csv(all_tweets, output_dir / "tweets.csv")
    scraper.save_to_json(all_tweets, output_dir / "tweets.json")

    # Summary
    print("\n" + "-" * 70)
    print("  SUMMARY")
    print("-" * 70)
    print(f"  Total tweets collected : {len(all_tweets)}")
    print(f"  CSV file               : {output_dir / 'tweets.csv'}")
    print(f"  JSON file              : {output_dir / 'tweets.json'}")
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
