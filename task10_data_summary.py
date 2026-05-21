"""
Task 10 – Data Summary & Visualization
=======================================
Generates a comprehensive statistical analysis of scraped Twitter/X data
collected during Tasks 8-9 and produces publication-ready visualizations.

Outputs:
    • Console   – full summary report
    • Text file – output/summary_report.txt
    • Plots     – plots/posts_per_keyword.png
                   plots/top_creators.png
                   plots/likes_vs_reposts.png
                   plots/content_distribution.png
"""

from __future__ import annotations

import random
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SCRAPER_DIR = BASE_DIR / "scraper_output"
PLOTS_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "output"

# Possible CSV locations (in priority order)
DATA_CANDIDATES = [
    SCRAPER_DIR / "tweets.csv",
    SCRAPER_DIR / "merged_tweets.csv",
]

# ── Sample-data generator ───────────────────────────────────────────────────

SAMPLE_KEYWORDS = [
    "machine learning", "deep learning", "NLP", "computer vision",
    "generative AI", "data science", "LLM", "transformers",
    "reinforcement learning", "AI ethics",
]

SAMPLE_USERNAMES = [
    "@ai_researcher", "@ml_engineer", "@data_guru", "@deep_learner",
    "@nlp_expert", "@cv_wizard", "@gen_ai_fan", "@llm_builder",
    "@rl_pioneer", "@ethicsAI", "@tech_today", "@neural_nerd",
    "@python_pro", "@kaggle_king", "@openai_watcher",
]


def _generate_sample_data(n: int = 120) -> pd.DataFrame:
    """Return a synthetic DataFrame that mimics scraped tweet data."""
    random.seed(42)
    rows = []
    for i in range(1, n + 1):
        keyword = random.choice(SAMPLE_KEYWORDS)
        username = random.choice(SAMPLE_USERNAMES)
        likes = random.randint(0, 5000)
        reposts = random.randint(0, int(likes * 0.6) + 1)
        rows.append(
            {
                "id": i,
                "username": username,
                "keyword": keyword,
                "text": f"Sample post about {keyword} #{keyword.replace(' ', '')}",
                "likes": likes,
                "reposts": reposts,
                "date": f"2025-05-{random.randint(1, 21):02d}",
            }
        )
    return pd.DataFrame(rows)


# ── Data loader ──────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """
    Try to load real scraped data.  Fall back to generated sample data
    if none of the expected CSV files exist.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least columns: username, keyword, likes, reposts.
    """
    for path in DATA_CANDIDATES:
        if path.is_file():
            print(f"✅  Loaded data from: {path.relative_to(BASE_DIR)}")
            df = pd.read_csv(path)
            # Normalise column names to lower-case for consistency
            df.columns = [c.strip().lower() for c in df.columns]
            return df

    # No real data found – generate sample data
    print(
        textwrap.dedent(
            """\
        ⚠️  No scraped data found!
            Expected one of:
              • scraper_output/tweets.csv
              • scraper_output/merged_tweets.csv

            Generating sample data for demonstration …
        """
        )
    )
    df = _generate_sample_data()
    # Persist so subsequent runs can reuse it
    SCRAPER_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SCRAPER_DIR / "tweets.csv"
    df.to_csv(sample_path, index=False)
    print(f"   Sample data saved to: {sample_path.relative_to(BASE_DIR)}\n")
    return df


# ── Analysis helpers ─────────────────────────────────────────────────────────


def compute_summary(df: pd.DataFrame) -> dict:
    """
    Compute all summary statistics from the DataFrame.

    Returns a dictionary with keys:
        total_posts, unique_creators, top_keywords, avg_likes, avg_reposts,
        top_posts, posts_per_keyword, posts_per_creator
    """
    # Ensure numeric columns
    for col in ("likes", "reposts"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    total_posts = len(df)
    unique_creators = df["username"].nunique() if "username" in df.columns else 0

    # Most common keywords (top 10)
    keyword_col = _resolve_keyword_column(df)
    keyword_counts = Counter(df[keyword_col].dropna())
    top_keywords = keyword_counts.most_common(10)

    # Engagement stats
    avg_likes = df["likes"].mean() if "likes" in df.columns else 0.0
    avg_reposts = df["reposts"].mean() if "reposts" in df.columns else 0.0

    # Top 5 posts by total engagement
    df["engagement"] = df.get("likes", 0) + df.get("reposts", 0)
    top_posts = df.nlargest(5, "engagement")

    # Per-keyword and per-creator post counts
    posts_per_keyword = df[keyword_col].value_counts()
    posts_per_creator = (
        df["username"].value_counts() if "username" in df.columns else pd.Series(dtype=int)
    )

    return {
        "total_posts": total_posts,
        "unique_creators": unique_creators,
        "top_keywords": top_keywords,
        "avg_likes": avg_likes,
        "avg_reposts": avg_reposts,
        "top_posts": top_posts,
        "posts_per_keyword": posts_per_keyword,
        "posts_per_creator": posts_per_creator,
        "keyword_col": keyword_col,
    }


def _resolve_keyword_column(df: pd.DataFrame) -> str:
    """Return the name of the column most likely holding keyword / category info."""
    for candidate in ("keyword", "category", "search_query", "query", "topic"):
        if candidate in df.columns:
            return candidate
    # Fall back to first string-like column
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return df.columns[0]


# ── Report builder ───────────────────────────────────────────────────────────


def build_report(stats: dict) -> str:
    """Format a human-readable text report from computed statistics."""
    lines: list[str] = []
    sep = "=" * 62

    lines.append(sep)
    lines.append("       TWITTER / X  DATA  SUMMARY  REPORT")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Total posts / videos collected : {stats['total_posts']}")
    lines.append(f"  Unique creators (usernames)    : {stats['unique_creators']}")
    lines.append(f"  Average likes per post         : {stats['avg_likes']:.1f}")
    lines.append(f"  Average reposts per post       : {stats['avg_reposts']:.1f}")
    lines.append("")

    lines.append("  ── Top 10 Keywords / Categories ──")
    for rank, (kw, count) in enumerate(stats["top_keywords"], 1):
        lines.append(f"    {rank:>2}. {kw:<30s}  ({count} posts)")
    lines.append("")

    lines.append("  ── Top 5 Posts by Engagement ──")
    top = stats["top_posts"]
    for idx, row in top.iterrows():
        username = row.get("username", "N/A")
        text_snippet = str(row.get("text", ""))[:80]
        likes = int(row.get("likes", 0))
        reposts = int(row.get("reposts", 0))
        lines.append(f"    • {username}  |  ❤ {likes}  🔁 {reposts}")
        lines.append(f"      \"{text_snippet}\"")
    lines.append("")

    lines.append(sep)
    lines.append("  Report generated by task10_data_summary.py")
    lines.append(sep)

    return "\n".join(lines)


# ── Visualizations ───────────────────────────────────────────────────────────


def create_visualizations(df: pd.DataFrame, stats: dict) -> list[Path]:
    """
    Create and save all visualisations.  Returns a list of saved file paths.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    saved: list[Path] = []

    # 1 ── Posts per keyword / category (bar chart) ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    kw_data = stats["posts_per_keyword"].head(10)
    kw_data.plot.barh(ax=ax, color=sns.color_palette("viridis", len(kw_data)))
    ax.set_xlabel("Number of Posts")
    ax.set_ylabel("Keyword / Category")
    ax.set_title("Posts per Keyword / Category")
    ax.invert_yaxis()
    fig.tight_layout()
    path = PLOTS_DIR / "posts_per_keyword.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊  Saved: {path.relative_to(BASE_DIR)}")

    # 2 ── Top 10 creators by post count (bar chart) ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    creator_data = stats["posts_per_creator"].head(10)
    creator_data.plot.barh(ax=ax, color=sns.color_palette("magma", len(creator_data)))
    ax.set_xlabel("Number of Posts")
    ax.set_ylabel("Creator (Username)")
    ax.set_title("Top 10 Creators by Post Count")
    ax.invert_yaxis()
    fig.tight_layout()
    path = PLOTS_DIR / "top_creators.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊  Saved: {path.relative_to(BASE_DIR)}")

    # 3 ── Likes vs Reposts scatter ─────────────────────────────────────────
    if "likes" in df.columns and "reposts" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        keyword_col = stats["keyword_col"]
        scatter_df = df.copy()
        # Colour by keyword (limited palette)
        categories = scatter_df[keyword_col].unique()
        palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
        for cat in categories:
            sub = scatter_df[scatter_df[keyword_col] == cat]
            ax.scatter(
                sub["likes"], sub["reposts"],
                label=cat, alpha=0.65, s=40,
                color=palette[cat], edgecolors="w", linewidths=0.3,
            )
        ax.set_xlabel("Likes")
        ax.set_ylabel("Reposts")
        ax.set_title("Likes vs Reposts (coloured by keyword)")
        ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
        fig.tight_layout()
        path = PLOTS_DIR / "likes_vs_reposts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  📊  Saved: {path.relative_to(BASE_DIR)}")

    # 4 ── Content distribution pie chart ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    kw_full = stats["posts_per_keyword"]
    ax.pie(
        kw_full.values,
        labels=kw_full.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel", len(kw_full)),
    )
    ax.set_title("Content Distribution by Category / Keyword")
    fig.tight_layout()
    path = PLOTS_DIR / "content_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  📊  Saved: {path.relative_to(BASE_DIR)}")

    return saved


# ── Main entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Run the full analysis pipeline."""
    print("\n🔍  Task 10 – Twitter / X Data Summary\n")

    # 1. Load data
    df = load_data()

    # 2. Compute summary statistics
    stats = compute_summary(df)

    # 3. Build & print the text report
    report = build_report(stats)
    print(report)

    # 4. Save report to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "summary_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n💾  Report saved to: {report_path.relative_to(BASE_DIR)}")

    # 5. Create visualisations
    print("\n📈  Generating visualisations …")
    saved_plots = create_visualizations(df, stats)
    print(f"\n✅  Done — {len(saved_plots)} plots saved to plots/\n")


if __name__ == "__main__":
    main()
