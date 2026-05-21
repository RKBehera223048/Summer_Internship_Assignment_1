"""
Task 10 – AI-Powered Data Analysis (LangChain + NVIDIA NIM)
============================================================
Uses LangChain with an NVIDIA NIM large-language model to produce an
AI-enhanced analysis of scraped Twitter/X data from Tasks 8-9.

The script:
    1. Loads (or generates sample) data
    2. Computes the same statistical metrics as task10_data_summary.py
    3. Builds context from those metrics and sends it to the LLM
    4. Merges the LLM response with the statistical report
    5. Saves visualisations to  plots/
    6. Saves enhanced report to  output/summary_report_langchain.txt

Environment
-----------
Set the NVIDIA_API_KEY environment variable before running:
    $env:NVIDIA_API_KEY = "nvapi-..."        # PowerShell
    set  NVIDIA_API_KEY=nvapi-...            # cmd
    export NVIDIA_API_KEY="nvapi-..."        # bash

Dependencies
------------
    pip install langchain-nvidia-ai-endpoints langchain-core pandas matplotlib seaborn
"""

from __future__ import annotations

import os
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
    Load real scraped data or fall back to generated sample data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns including username, keyword, likes, reposts.
    """
    for path in DATA_CANDIDATES:
        if path.is_file():
            print(f"✅  Loaded data from: {path.relative_to(BASE_DIR)}")
            df = pd.read_csv(path)
            df.columns = [c.strip().lower() for c in df.columns]
            return df

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
    SCRAPER_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SCRAPER_DIR / "tweets.csv"
    df.to_csv(sample_path, index=False)
    print(f"   Sample data saved to: {sample_path.relative_to(BASE_DIR)}\n")
    return df


# ── Analysis helpers ─────────────────────────────────────────────────────────


def _resolve_keyword_column(df: pd.DataFrame) -> str:
    """Return the name of the column most likely holding keyword / category info."""
    for candidate in ("keyword", "category", "search_query", "query", "topic"):
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if df[col].dtype == object:
            return col
    return df.columns[0]


def compute_summary(df: pd.DataFrame) -> dict:
    """
    Compute all summary statistics from the DataFrame.

    Returns a dictionary with keys:
        total_posts, unique_creators, top_keywords, avg_likes, avg_reposts,
        top_posts, posts_per_keyword, posts_per_creator, keyword_col
    """
    for col in ("likes", "reposts"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    total_posts = len(df)
    unique_creators = df["username"].nunique() if "username" in df.columns else 0

    keyword_col = _resolve_keyword_column(df)
    keyword_counts = Counter(df[keyword_col].dropna())
    top_keywords = keyword_counts.most_common(10)

    avg_likes = df["likes"].mean() if "likes" in df.columns else 0.0
    avg_reposts = df["reposts"].mean() if "reposts" in df.columns else 0.0

    df["engagement"] = df.get("likes", 0) + df.get("reposts", 0)
    top_posts = df.nlargest(5, "engagement")

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


# ── Report builders ─────────────────────────────────────────────────────────


def build_stats_section(stats: dict) -> str:
    """Format the statistical section of the report."""
    lines: list[str] = []
    sep = "=" * 62

    lines.append(sep)
    lines.append("       STATISTICAL  SUMMARY")
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
    for _, row in stats["top_posts"].iterrows():
        username = row.get("username", "N/A")
        text_snippet = str(row.get("text", ""))[:80]
        likes = int(row.get("likes", 0))
        reposts = int(row.get("reposts", 0))
        lines.append(f"    • {username}  |  ❤ {likes}  🔁 {reposts}")
        lines.append(f"      \"{text_snippet}\"")
    lines.append("")
    return "\n".join(lines)


def _build_llm_context(stats: dict) -> str:
    """Prepare a concise textual context to pass to the LLM."""
    kw_summary = ", ".join(f"{kw} ({c})" for kw, c in stats["top_keywords"])
    top_creators = ", ".join(
        f"{name} ({count})" for name, count in stats["posts_per_creator"].head(5).items()
    )

    top_post_lines: list[str] = []
    for _, row in stats["top_posts"].iterrows():
        top_post_lines.append(
            f"  - @{row.get('username','?')}: {row.get('likes',0)} likes, "
            f"{row.get('reposts',0)} reposts — \"{str(row.get('text',''))[:100]}\""
        )

    return textwrap.dedent(f"""\
        Twitter / X Data Collection Summary
        ------------------------------------
        Total posts collected     : {stats['total_posts']}
        Unique creators           : {stats['unique_creators']}
        Average likes per post    : {stats['avg_likes']:.1f}
        Average reposts per post  : {stats['avg_reposts']:.1f}

        Top 10 keywords (with count): {kw_summary}
        Top 5 creators  (with count): {top_creators}

        Top 5 performing posts:
        {chr(10).join(top_post_lines)}
    """)


# ── LangChain / NVIDIA NIM ──────────────────────────────────────────────────


def get_llm_analysis(context: str) -> str | None:
    """
    Call NVIDIA NIM via LangChain and return the AI-generated analysis.

    Returns None if the API key is missing or the call fails.
    """
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print(
            textwrap.dedent(
                """\
            ⚠️  NVIDIA_API_KEY environment variable is not set.
                To enable AI-powered analysis, set it before running:

                  PowerShell :  $env:NVIDIA_API_KEY = "nvapi-..."
                  cmd        :  set  NVIDIA_API_KEY=nvapi-...
                  bash       :  export NVIDIA_API_KEY="nvapi-..."

                Proceeding in fallback mode (statistics only) …
            """
            )
        )
        return None

    # Import LangChain components (deferred so the script still works
    # without langchain installed when running in fallback mode)
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        print(f"⚠️  LangChain import failed: {exc}")
        print("    Install with:  pip install langchain-nvidia-ai-endpoints langchain-core")
        print("    Proceeding in fallback mode (statistics only) …\n")
        return None

    # Initialise the NVIDIA NIM model
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        api_key=api_key,
        temperature=0.7,
        max_tokens=2048,
    )

    system_prompt = (
        "You are a social-media analytics expert.  Given the statistical summary "
        "of a Twitter/X data collection, produce a detailed but concise analysis "
        "covering:\n"
        "1. **Trends** – key patterns in keywords, topics, and posting volume.\n"
        "2. **Content quality** – assess variety and depth of collected content.\n"
        "3. **Engagement patterns** – analyse likes vs reposts, top performers.\n"
        "4. **Recommendations** – actionable content-strategy suggestions.\n\n"
        "Use clear headings and bullet points.  Keep total length under 600 words."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Here is the data summary:\n\n{context}"),
    ]

    try:
        print("🤖  Querying NVIDIA NIM (meta/llama-3.1-8b-instruct) …")
        response = llm.invoke(messages)
        print("✅  AI analysis received.\n")
        return response.content
    except Exception as exc:
        print(f"⚠️  LLM API call failed: {exc}")
        print("    Proceeding in fallback mode (statistics only) …\n")
        return None


# ── Visualizations (same set as task10_data_summary.py) ──────────────────────


def create_visualizations(df: pd.DataFrame, stats: dict) -> list[Path]:
    """Create and save all visualisations.  Returns list of saved paths."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")
    saved: list[Path] = []

    # 1 ── Posts per keyword (bar chart) ─────────────────────────────────────
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

    # 2 ── Top 10 creators (bar chart) ──────────────────────────────────────
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
        categories = df[keyword_col].unique()
        palette = dict(zip(categories, sns.color_palette("husl", len(categories))))
        for cat in categories:
            sub = df[df[keyword_col] == cat]
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

    # 4 ── Content distribution pie ─────────────────────────────────────────
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
    """Run the full AI-enhanced analysis pipeline."""
    print("\n🔍  Task 10 – AI-Powered Data Analysis (LangChain + NVIDIA NIM)\n")

    # 1. Load data
    df = load_data()

    # 2. Compute summary statistics
    stats = compute_summary(df)

    # 3. Build statistical section
    stats_text = build_stats_section(stats)
    print(stats_text)

    # 4. Attempt LLM-powered analysis
    llm_context = _build_llm_context(stats)
    ai_analysis = get_llm_analysis(llm_context)

    # 5. Compose final report
    sep = "=" * 62
    report_parts: list[str] = [
        sep,
        "   TWITTER / X  DATA  ANALYSIS  (LangChain + NVIDIA NIM)",
        sep,
        "",
        stats_text,
    ]

    if ai_analysis:
        report_parts.extend(
            [
                sep,
                "       AI-GENERATED  ANALYSIS",
                sep,
                "",
                ai_analysis,
                "",
            ]
        )
        print(f"\n{sep}")
        print("       AI-GENERATED  ANALYSIS")
        print(sep)
        print(ai_analysis)
    else:
        fallback_note = (
            "\n[AI analysis unavailable — running in fallback (statistics-only) mode.\n"
            " Set NVIDIA_API_KEY and install langchain-nvidia-ai-endpoints to enable.]\n"
        )
        report_parts.append(fallback_note)
        print(fallback_note)

    report_parts.extend(
        [
            sep,
            "  Report generated by task10_langchain_analysis.py",
            sep,
        ]
    )

    full_report = "\n".join(report_parts)

    # 6. Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "summary_report_langchain.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\n💾  Report saved to: {report_path.relative_to(BASE_DIR)}")

    # 7. Create visualisations
    print("\n📈  Generating visualisations …")
    saved_plots = create_visualizations(df, stats)
    print(f"\n✅  Done — {len(saved_plots)} plots saved to plots/\n")


if __name__ == "__main__":
    main()
