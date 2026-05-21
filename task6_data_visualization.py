"""
Task 6: Data Visualization
===========================
Creates at least 6 publication-quality visualizations from the cleaned
Digital Burnout and Productivity Analytics dataset.

Plots generated:
  1. Histogram   – Distribution of a key numeric column
  2. Scatter     – Relationship between two numeric variables
  3. Heatmap     – Correlation matrix of all numeric columns
  4. Bar Chart   – Mean of a numeric measure grouped by a category
  5. Box Plot    – Distribution of a numeric column across categories
  6. Pie Chart   – Proportions of a categorical variable (bonus)

All plots are saved to the `plots/` directory as PNG files.
"""

from pathlib import Path
from textwrap import wrap

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"

CLEANED_CSV = DATA_DIR / "cleaned_dataset.csv"
RAW_CSV = DATA_DIR / "digital_burnout_productivity_dataset_5M.csv"


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------
def _apply_style() -> None:
    """Set a consistent, attractive plot style."""
    available = plt.style.available
    preferred = [
        "seaborn-v0_8-darkgrid",
        "seaborn-darkgrid",
        "seaborn-v0_8",
        "seaborn",
        "ggplot",
    ]
    for s in preferred:
        if s in available:
            plt.style.use(s)
            print(f"[Style] Using matplotlib style: '{s}'")
            return
    print("[Style] Using default matplotlib style")


# ---------------------------------------------------------------------------
# Column auto-detection helpers
# ---------------------------------------------------------------------------
def _find_col(df: pd.DataFrame, *candidates: str, dtype: str = "any") -> str | None:
    """
    Return the first column in *candidates* that exists in *df*.
    If dtype is 'numeric', only return numeric columns.
    If dtype is 'categorical', only return object/category columns.
    """
    for c in candidates:
        for col in df.columns:
            if c.lower() == col.lower():
                if dtype == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                if dtype == "categorical" and pd.api.types.is_numeric_dtype(df[col]):
                    continue
                return col
    return None


def _get_numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return all numeric column names."""
    return df.select_dtypes(include="number").columns.tolist()


def _get_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Return all object / category column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _pick_best_numeric(df: pd.DataFrame, *candidates: str) -> str:
    """Pick the first matching numeric column, or fall back to the first numeric col."""
    col = _find_col(df, *candidates, dtype="numeric")
    if col:
        return col
    numerics = _get_numeric_cols(df)
    # Exclude id-like columns
    numerics = [c for c in numerics if "id" not in c.lower()]
    return numerics[0] if numerics else _get_numeric_cols(df)[0]


def _pick_best_categorical(df: pd.DataFrame, *candidates: str,
                           max_unique: int = 15) -> str:
    """Pick a categorical column with a reasonable number of unique values."""
    col = _find_col(df, *candidates, dtype="categorical")
    if col and df[col].nunique() <= max_unique:
        return col
    cats = _get_categorical_cols(df)
    for c in cats:
        if df[c].nunique() <= max_unique:
            return c
    # If all have many uniques, return first candidate anyway
    return cats[0] if cats else None


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_histogram(df: pd.DataFrame) -> Path | None:
    """1. Histogram — distribution of a key numeric variable."""
    col = _pick_best_numeric(
        df, "daily_screen_time", "screen_time", "total_screen_time",
        "stress_level", "sleep_hours",
    )
    print(f"\n[Plot 1] Histogram of '{col}'")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[col].dropna(), bins=50, color="#4C72B0", edgecolor="white",
            alpha=0.85, label=col)
    ax.set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=14,
                 fontweight="bold")
    ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)

    # Add mean/median reference lines
    mean_val = df[col].mean()
    med_val = df[col].median()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.3,
               label=f"Mean = {mean_val:.2f}")
    ax.axvline(med_val, color="green", linestyle="-.", linewidth=1.3,
               label=f"Median = {med_val:.2f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = PLOTS_DIR / "01_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_scatter(df: pd.DataFrame) -> Path | None:
    """2. Scatter plot — relationship between two numeric variables."""
    x_col = _pick_best_numeric(
        df, "daily_screen_time", "screen_time", "total_screen_time",
        "social_media_time",
    )
    y_col = _pick_best_numeric(
        df, "productivity_score", "task_completion_rate",
        "burnout_score", "stress_level",
    )
    # Make sure x and y are different
    if x_col == y_col:
        numerics = [c for c in _get_numeric_cols(df) if c != x_col and "id" not in c.lower()]
        y_col = numerics[0] if numerics else x_col

    print(f"[Plot 2] Scatter: '{x_col}' vs '{y_col}'")

    # Sample to avoid overcrowding
    sample = df.sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        sample[x_col], sample[y_col],
        c=sample[y_col], cmap="viridis", alpha=0.5, s=12, edgecolors="none",
    )
    fig.colorbar(scatter, ax=ax, label=y_col.replace("_", " ").title())
    ax.set_title(
        f"{x_col.replace('_',' ').title()} vs {y_col.replace('_',' ').title()}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)

    plt.tight_layout()
    path = PLOTS_DIR / "02_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap(df: pd.DataFrame) -> Path | None:
    """3. Heatmap — correlation matrix of numeric columns."""
    numerics = [c for c in _get_numeric_cols(df) if "id" not in c.lower()]
    if len(numerics) < 2:
        print("[Plot 3] Not enough numeric columns for a heatmap. Skipping.")
        return None

    # Limit columns for readability
    if len(numerics) > 20:
        numerics = numerics[:20]

    print(f"[Plot 3] Correlation heatmap ({len(numerics)} numeric columns)")

    corr = df[numerics].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 7},
    )
    ax.set_title("Correlation Matrix — Numeric Features", fontsize=14,
                 fontweight="bold")
    # Wrap long tick labels
    ax.set_xticklabels(
        ["\n".join(wrap(t.get_text().replace("_", " "), 15))
         for t in ax.get_xticklabels()],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_yticklabels(
        ["\n".join(wrap(t.get_text().replace("_", " "), 15))
         for t in ax.get_yticklabels()],
        fontsize=8,
    )

    plt.tight_layout()
    path = PLOTS_DIR / "03_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bar_chart(df: pd.DataFrame) -> Path | None:
    """4. Bar chart — mean of a numeric measure grouped by a category."""
    cat_col = _pick_best_categorical(
        df, "work_setting", "burnout_risk", "sleep_quality",
        "productivity_category", "occupation", "gender",
    )
    num_col = _pick_best_numeric(
        df, "productivity_score", "daily_screen_time",
        "stress_level", "task_completion_rate",
    )

    if cat_col is None:
        print("[Plot 4] No suitable categorical column found. Skipping.")
        return None

    print(f"[Plot 4] Bar chart: mean '{num_col}' by '{cat_col}'")

    grouped = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", n_colors=len(grouped))
    bars = ax.bar(grouped.index.astype(str), grouped.values, color=colors,
                  edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.set_title(
        f"Mean {num_col.replace('_',' ').title()} by {cat_col.replace('_',' ').title()}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(cat_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(f"Mean {num_col.replace('_', ' ').title()}", fontsize=12)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = PLOTS_DIR / "04_bar_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_boxplot(df: pd.DataFrame) -> Path | None:
    """5. Box plot — numeric distribution across categories."""
    cat_col = _pick_best_categorical(
        df, "burnout_risk", "sleep_quality", "work_setting",
        "productivity_category", "gender",
    )
    num_col = _pick_best_numeric(
        df, "stress_level", "mental_fatigue_score",
        "daily_screen_time", "productivity_score",
    )

    if cat_col is None:
        print("[Plot 5] No suitable categorical column. Skipping.")
        return None

    print(f"[Plot 5] Box plot: '{num_col}' across '{cat_col}'")

    fig, ax = plt.subplots(figsize=(10, 6))
    order = sorted(df[cat_col].dropna().unique())
    sns.boxplot(data=df, x=cat_col, y=num_col, order=order,
                palette="pastel", ax=ax, linewidth=1.2)
    ax.set_title(
        f"{num_col.replace('_',' ').title()} Distribution by {cat_col.replace('_',' ').title()}",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel(cat_col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(num_col.replace("_", " ").title(), fontsize=12)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = PLOTS_DIR / "05_boxplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pie_chart(df: pd.DataFrame) -> Path | None:
    """6. Pie chart — proportions of a categorical variable (bonus)."""
    cat_col = _pick_best_categorical(
        df, "burnout_risk", "productivity_category",
        "work_setting", "sleep_quality", "gender",
        max_unique=10,
    )
    if cat_col is None:
        print("[Plot 6] No suitable categorical column for pie chart. Skipping.")
        return None

    print(f"[Plot 6] Pie chart: proportions of '{cat_col}'")

    counts = df[cat_col].value_counts()
    colors = sns.color_palette("Set3", n_colors=len(counts))

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index.astype(str),
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.85,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")

    # Draw a centre circle for a donut effect
    centre = plt.Circle((0, 0), 0.60, fc="white")
    ax.add_patch(centre)

    ax.set_title(
        f"Distribution of {cat_col.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.legend(counts.index.astype(str), title=cat_col.replace("_", " ").title(),
              loc="lower right", fontsize=9)

    plt.tight_layout()
    path = PLOTS_DIR / "06_pie_chart.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate all visualizations and save to plots/."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Task 6 – Data Visualization                                  ║")
    print("║   Dataset: Digital Burnout and Productivity Analytics           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    if CLEANED_CSV.exists():
        print(f"\n[Load] Using cleaned dataset: {CLEANED_CSV}")
        df = pd.read_csv(CLEANED_CSV)
    elif RAW_CSV.exists():
        print(f"\n[Load] Cleaned CSV not found. Using raw dataset: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV, nrows=50_000)
    else:
        print("[Error] No dataset found. Please run task5_data_cleaning.py first.")
        return

    print(f"[Load] Loaded {len(df):,} rows, {len(df.columns)} columns.")
    print(f"[Load] Numeric columns: {len(_get_numeric_cols(df))}")
    print(f"[Load] Categorical columns: {len(_get_categorical_cols(df))}")

    _apply_style()

    # Generate all plots
    plot_functions = [
        ("Histogram", plot_histogram),
        ("Scatter Plot", plot_scatter),
        ("Heatmap", plot_heatmap),
        ("Bar Chart", plot_bar_chart),
        ("Box Plot", plot_boxplot),
        ("Pie Chart (Bonus)", plot_pie_chart),
    ]

    created: list[tuple[str, Path]] = []
    failed: list[str] = []

    for name, func in plot_functions:
        try:
            path = func(df)
            if path and path.exists():
                created.append((name, path))
            else:
                failed.append(name)
        except Exception as exc:
            print(f"[Error] {name} failed: {exc}")
            failed.append(name)

    # Summary
    print("\n" + "=" * 70)
    print("  VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"\n  Successfully created: {len(created)} / {len(plot_functions)} plots\n")
    for name, path in created:
        size_kb = path.stat().st_size / 1024
        print(f"    ✓ {name:<22} → {path.name}  ({size_kb:.1f} KB)")
    if failed:
        print(f"\n  Failed: {len(failed)}")
        for name in failed:
            print(f"    ✗ {name}")
    print(f"\n  All plots saved to: {PLOTS_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
