"""
Task 7: Insights Report Generator
===================================
Programmatically generates a comprehensive Markdown report analyzing the
Digital Burnout and Productivity Analytics dataset.

Report sections:
  1. Dataset Overview (rows, columns, types)
  2. Key Statistics (means, medians, std devs)
  3. Correlation Analysis (strongest correlations)
  4. Distribution Patterns (skewness, outliers via IQR)
  5. Data Quality Issues (missing values, duplicates)
  6. Possible Issues in the Dataset (synthetic patterns)
  7. Key Conclusions and Trends

Output: output/insights_report.md
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

CLEANED_CSV = DATA_DIR / "cleaned_dataset.csv"
RAW_CSV = DATA_DIR / "digital_burnout_productivity_dataset_5M.csv"
REPORT_FILE = OUTPUT_DIR / "insights_report.md"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, str]:
    """Load the best available dataset. Returns (df, source_label)."""
    if CLEANED_CSV.exists():
        print(f"[Load] Using cleaned dataset: {CLEANED_CSV}")
        return pd.read_csv(CLEANED_CSV), "cleaned_dataset.csv"
    elif RAW_CSV.exists():
        print(f"[Load] Using raw dataset: {RAW_CSV}")
        return pd.read_csv(RAW_CSV, nrows=50_000), RAW_CSV.name
    else:
        raise FileNotFoundError(
            "No dataset found. Please run task5_data_cleaning.py first."
        )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _get_numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in cols if "id" not in c.lower()]


def _get_categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _iqr_outliers(series: pd.Series) -> dict:
    """Detect outliers using the IQR method. Returns stats dict."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "q1": q1, "q3": q3, "iqr": iqr,
        "lower_bound": lower, "upper_bound": upper,
        "n_outliers": len(outliers),
        "pct_outliers": len(outliers) / len(series) * 100 if len(series) > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def section_overview(df: pd.DataFrame, source: str) -> str:
    """Section 1: Dataset Overview."""
    num_cols = _get_numeric_cols(df)
    cat_cols = _get_categorical_cols(df)

    lines = [
        "## 1. Dataset Overview\n",
        f"| Property | Value |",
        f"|---|---|",
        f"| **Source File** | `{source}` |",
        f"| **Rows** | {df.shape[0]:,} |",
        f"| **Columns** | {df.shape[1]} |",
        f"| **Numeric Columns** | {len(num_cols)} |",
        f"| **Categorical Columns** | {len(cat_cols)} |",
        f"| **Memory Usage** | {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB |",
        "",
        "### Column Inventory\n",
        "| # | Column Name | Data Type | Non-Null Count | Unique Values |",
        "|---|---|---|---|---|",
    ]

    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        nunique = df[col].nunique()
        lines.append(f"| {i} | `{col}` | {dtype} | {non_null:,} | {nunique:,} |")

    lines.append("")
    return "\n".join(lines)


def section_key_statistics(df: pd.DataFrame) -> str:
    """Section 2: Key Statistics."""
    num_cols = _get_numeric_cols(df)

    lines = [
        "## 2. Key Statistics\n",
        "### Numeric Column Summary\n",
        "| Column | Mean | Median | Std Dev | Min | Max |",
        "|---|---|---|---|---|---|",
    ]

    for col in num_cols:
        s = df[col]
        lines.append(
            f"| `{col}` | {s.mean():.3f} | {s.median():.3f} | "
            f"{s.std():.3f} | {s.min():.3f} | {s.max():.3f} |"
        )

    # Categorical summaries
    cat_cols = _get_categorical_cols(df)
    if cat_cols:
        lines.append("\n### Categorical Column Summary\n")
        lines.append("| Column | Unique Values | Most Common | Frequency |")
        lines.append("|---|---|---|---|")
        for col in cat_cols:
            nunique = df[col].nunique()
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            mode_count = df[col].value_counts().iloc[0] if nunique > 0 else 0
            lines.append(f"| `{col}` | {nunique} | {mode_val} | {mode_count:,} |")

    lines.append("")
    return "\n".join(lines)


def section_correlation(df: pd.DataFrame) -> str:
    """Section 3: Correlation Analysis."""
    num_cols = _get_numeric_cols(df)

    if len(num_cols) < 2:
        return "## 3. Correlation Analysis\n\nInsufficient numeric columns.\n"

    corr = df[num_cols].corr()

    # Flatten and sort correlations (exclude self-correlation)
    pairs = []
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if i < j:
                r = corr.loc[c1, c2]
                pairs.append((c1, c2, r, abs(r)))

    pairs.sort(key=lambda x: x[3], reverse=True)

    lines = [
        "## 3. Correlation Analysis\n",
        "### Top 15 Strongest Correlations\n",
        "| Rank | Feature 1 | Feature 2 | Correlation | Strength |",
        "|---|---|---|---|---|",
    ]

    for rank, (c1, c2, r, ar) in enumerate(pairs[:15], 1):
        strength = (
            "🔴 Very Strong" if ar >= 0.8 else
            "🟠 Strong" if ar >= 0.6 else
            "🟡 Moderate" if ar >= 0.4 else
            "🟢 Weak" if ar >= 0.2 else
            "⚪ Very Weak"
        )
        lines.append(f"| {rank} | `{c1}` | `{c2}` | {r:+.4f} | {strength} |")

    # Note weakest
    lines.append("\n### Weakest Correlations (near zero)\n")
    lines.append("| Feature 1 | Feature 2 | Correlation |")
    lines.append("|---|---|---|")
    for c1, c2, r, ar in pairs[-5:]:
        lines.append(f"| `{c1}` | `{c2}` | {r:+.4f} |")

    # Key findings
    lines.append("\n### Key Correlation Findings\n")
    if pairs:
        top = pairs[0]
        lines.append(
            f"- **Strongest correlation**: `{top[0]}` and `{top[1]}` "
            f"(r = {top[2]:+.4f})"
        )
        # Count strong correlations
        strong = [p for p in pairs if p[3] >= 0.6]
        lines.append(f"- **Number of strong correlations** (|r| ≥ 0.6): {len(strong)}")
        moderate = [p for p in pairs if 0.4 <= p[3] < 0.6]
        lines.append(f"- **Number of moderate correlations** (0.4 ≤ |r| < 0.6): {len(moderate)}")

    lines.append("")
    return "\n".join(lines)


def section_distributions(df: pd.DataFrame) -> str:
    """Section 4: Distribution Patterns."""
    num_cols = _get_numeric_cols(df)

    lines = [
        "## 4. Distribution Patterns\n",
        "### Skewness Analysis\n",
        "| Column | Skewness | Kurtosis | Interpretation |",
        "|---|---|---|---|",
    ]

    for col in num_cols:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        if abs(skew) < 0.5:
            interp = "Approximately symmetric"
        elif skew > 0:
            interp = f"Right-skewed ({'moderately' if skew < 1 else 'highly'})"
        else:
            interp = f"Left-skewed ({'moderately' if abs(skew) < 1 else 'highly'})"
        lines.append(f"| `{col}` | {skew:.3f} | {kurt:.3f} | {interp} |")

    # Outlier analysis via IQR
    lines.append("\n### Outlier Detection (IQR Method)\n")
    lines.append("| Column | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outliers | % |")
    lines.append("|---|---|---|---|---|---|---|---|")

    outlier_summary = []
    for col in num_cols:
        stats = _iqr_outliers(df[col].dropna())
        outlier_summary.append((col, stats))
        lines.append(
            f"| `{col}` | {stats['q1']:.2f} | {stats['q3']:.2f} | "
            f"{stats['iqr']:.2f} | {stats['lower_bound']:.2f} | "
            f"{stats['upper_bound']:.2f} | {stats['n_outliers']:,} | "
            f"{stats['pct_outliers']:.1f}% |"
        )

    # Highlight columns with many outliers
    high_outliers = [(c, s) for c, s in outlier_summary if s["pct_outliers"] > 5]
    if high_outliers:
        lines.append("\n> [!WARNING]")
        lines.append("> **Columns with >5% outliers:**")
        for c, s in high_outliers:
            lines.append(f"> - `{c}`: {s['n_outliers']:,} outliers ({s['pct_outliers']:.1f}%)")

    lines.append("")
    return "\n".join(lines)


def section_data_quality(df: pd.DataFrame) -> str:
    """Section 5: Data Quality Issues."""
    lines = [
        "## 5. Data Quality Assessment\n",
        "### Missing Values\n",
    ]

    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    total_cells = df.shape[0] * df.shape[1]

    if total_nulls == 0:
        lines.append(
            "✅ **No missing values** detected in the dataset. "
            "The cleaning pipeline has successfully imputed all gaps.\n"
        )
    else:
        lines.append("| Column | Missing Count | Missing % |")
        lines.append("|---|---|---|")
        for col in null_counts[null_counts > 0].index:
            pct = null_counts[col] / len(df) * 100
            lines.append(f"| `{col}` | {null_counts[col]:,} | {pct:.2f}% |")
        lines.append(f"\n**Total missing cells**: {total_nulls:,} / "
                      f"{total_cells:,} ({total_nulls/total_cells*100:.2f}%)\n")

    # Duplicates
    n_dups = df.duplicated().sum()
    lines.append("### Duplicate Rows\n")
    if n_dups == 0:
        lines.append("✅ **No duplicate rows** found in the dataset.\n")
    else:
        lines.append(f"⚠️ **{n_dups:,} duplicate rows** detected "
                      f"({n_dups/len(df)*100:.2f}% of data).\n")

    # Data type consistency
    lines.append("### Data Type Summary\n")
    lines.append("| Data Type | Count |")
    lines.append("|---|---|")
    for dtype, count in df.dtypes.value_counts().items():
        lines.append(f"| {dtype} | {count} |")

    lines.append("")
    return "\n".join(lines)


def section_possible_issues(df: pd.DataFrame) -> str:
    """Section 6: Possible Issues in the Dataset."""
    lines = [
        "## 6. Possible Issues in the Dataset\n",
    ]

    num_cols = _get_numeric_cols(df)
    issues_found = []

    # Check for suspiciously uniform distributions
    for col in num_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        # Check if values are suspiciously evenly distributed
        cv = s.std() / s.mean() if s.mean() != 0 else 0
        if 0.55 < cv < 0.60:
            issues_found.append(
                f"- `{col}`: Coefficient of variation ({cv:.3f}) suggests "
                f"possible uniform random generation"
            )

    # Check for perfect integer ranges
    for col in num_cols:
        s = df[col].dropna()
        if s.dtype in [np.int64, np.int32, int]:
            if s.min() == 1 and s.max() == 10:
                issues_found.append(
                    f"- `{col}`: Perfect 1-10 integer range — may indicate "
                    f"synthetic Likert-scale data"
                )

    # Check for no correlation between logically related features
    corr = df[num_cols].corr() if len(num_cols) >= 2 else pd.DataFrame()
    low_corr_pairs = []
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if i < j and abs(corr.loc[c1, c2]) < 0.05:
                # Check if names suggest they should be related
                related_keywords = [
                    ("screen", "productivity"), ("stress", "burnout"),
                    ("sleep", "fatigue"), ("screen", "burnout"),
                ]
                for k1, k2 in related_keywords:
                    if (k1 in c1.lower() and k2 in c2.lower()) or \
                       (k2 in c1.lower() and k1 in c2.lower()):
                        low_corr_pairs.append((c1, c2, corr.loc[c1, c2]))

    if low_corr_pairs:
        issues_found.append("\n**Unexpected lack of correlation between related features:**")
        for c1, c2, r in low_corr_pairs:
            issues_found.append(
                f"- `{c1}` vs `{c2}`: r = {r:.4f} (expected stronger relationship)"
            )

    # Check for unrealistic value ranges
    for col in num_cols:
        s = df[col].dropna()
        if "time" in col.lower() or "hours" in col.lower():
            if s.max() > 24:
                issues_found.append(
                    f"- `{col}`: Maximum value {s.max():.1f} exceeds 24 hours"
                )

    if issues_found:
        lines.append("> [!NOTE]")
        lines.append("> This is a **synthetic dataset** created for educational purposes.")
        lines.append("> The following patterns are consistent with synthetic data generation:\n")
        lines.extend(issues_found)
    else:
        lines.append(
            "No major data quality issues or unrealistic patterns detected. "
            "However, note that this is a synthetic dataset designed for "
            "educational and analytical purposes.\n"
        )

    lines.append("")
    return "\n".join(lines)


def section_conclusions(df: pd.DataFrame) -> str:
    """Section 7: Key Conclusions and Trends."""
    num_cols = _get_numeric_cols(df)
    cat_cols = _get_categorical_cols(df)

    lines = [
        "## 7. Key Conclusions and Trends\n",
    ]

    # Auto-generate insights based on data
    insights = []

    # Insight: column with highest variability
    if num_cols:
        cvs = {}
        for col in num_cols:
            s = df[col].dropna()
            if s.mean() != 0:
                cvs[col] = s.std() / abs(s.mean())
        if cvs:
            most_variable = max(cvs, key=cvs.get)
            insights.append(
                f"**Most variable feature**: `{most_variable}` "
                f"(CV = {cvs[most_variable]:.3f}), suggesting high diversity "
                f"in this metric across the population."
            )

    # Insight: strongest correlation finding
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        pairs = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j:
                    pairs.append((c1, c2, corr.loc[c1, c2], abs(corr.loc[c1, c2])))
        pairs.sort(key=lambda x: x[3], reverse=True)
        if pairs:
            top = pairs[0]
            direction = "positive" if top[2] > 0 else "negative"
            insights.append(
                f"**Strongest relationship**: `{top[0]}` and `{top[1]}` show a "
                f"{direction} correlation (r = {top[2]:+.4f}), indicating these "
                f"features tend to {'increase' if top[2] > 0 else 'decrease'} together."
            )

    # Insight: skewed distributions
    skewed = []
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            skewed.append((col, skew))
    if skewed:
        skewed.sort(key=lambda x: abs(x[1]), reverse=True)
        top_skew = skewed[0]
        insights.append(
            f"**Most skewed distribution**: `{top_skew[0]}` "
            f"(skewness = {top_skew[1]:.3f}). "
            f"{'Right-skewed distributions suggest a long tail of high values.' if top_skew[1] > 0 else 'Left-skewed distributions suggest most values are concentrated at the higher end.'}"
        )

    # Insight: categorical distribution balance
    for col in cat_cols:
        vc = df[col].value_counts(normalize=True)
        if vc.max() > 0.5:
            insights.append(
                f"**Imbalanced category**: `{col}` — the value '{vc.idxmax()}' "
                f"accounts for {vc.max()*100:.1f}% of observations, which may "
                f"affect model performance if used as a target variable."
            )

    # Insight: overall data quality
    null_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    dup_pct = df.duplicated().sum() / len(df) * 100
    insights.append(
        f"**Data Quality Score**: The dataset has {null_pct:.2f}% missing values "
        f"and {dup_pct:.2f}% duplicates after cleaning, indicating "
        f"{'excellent' if null_pct < 1 and dup_pct < 1 else 'good' if null_pct < 5 else 'moderate'} "
        f"data quality."
    )

    # Format insights
    for i, insight in enumerate(insights, 1):
        lines.append(f"{i}. {insight}\n")

    # Summary box
    lines.append("> [!TIP]")
    lines.append("> **Recommendations for further analysis:**")
    lines.append("> - Apply feature scaling before ML modeling")
    lines.append("> - Consider dimensionality reduction (PCA) given the number of features")
    lines.append("> - Use stratified sampling if target variables are imbalanced")
    lines.append("> - Investigate causal relationships between screen time and productivity")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def generate_report(df: pd.DataFrame, source: str) -> str:
    """Assemble the full Markdown report."""
    header = (
        "# 📊 Digital Burnout & Productivity Analytics — Insights Report\n\n"
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"**Source**: `{source}`  \n"
        f"**Rows analyzed**: {len(df):,}  \n"
        f"**Columns**: {len(df.columns)}  \n\n"
        "---\n\n"
    )

    sections = [
        header,
        section_overview(df, source),
        section_key_statistics(df),
        section_correlation(df),
        section_distributions(df),
        section_data_quality(df),
        section_possible_issues(df),
        section_conclusions(df),
    ]

    # Footer
    sections.append(
        "---\n\n"
        "*Report auto-generated by `task7_insights_report.py`*\n"
    )

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_console_summary(df: pd.DataFrame) -> None:
    """Print a condensed summary to the console."""
    num_cols = _get_numeric_cols(df)

    print("\n" + "=" * 70)
    print("  CONSOLE SUMMARY")
    print("=" * 70)

    print(f"\n  Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(_get_categorical_cols(df))}")
    print(f"  Missing values: {df.isnull().sum().sum():,}")
    print(f"  Duplicates: {df.duplicated().sum():,}")

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        # Find strongest correlation
        pairs = []
        for i, c1 in enumerate(num_cols):
            for j, c2 in enumerate(num_cols):
                if i < j:
                    pairs.append((c1, c2, abs(corr.loc[c1, c2])))
        if pairs:
            pairs.sort(key=lambda x: x[2], reverse=True)
            top = pairs[0]
            print(f"\n  Strongest correlation: {top[0]} ↔ {top[1]} (|r| = {top[2]:.4f})")

    # Top skewed columns
    skewed = [(col, df[col].skew()) for col in num_cols if abs(df[col].skew()) > 1]
    if skewed:
        skewed.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Most skewed column: {skewed[0][0]} (skewness = {skewed[0][1]:.3f})")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate the insights report and save to output/."""
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   Task 7 – Insights Report Generator                           ║")
    print("║   Dataset: Digital Burnout and Productivity Analytics           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df, source = load_data()
    print(f"[Load] {len(df):,} rows, {len(df.columns)} columns from '{source}'")

    # Generate report
    print("\n[Report] Generating insights report …")
    report_md = generate_report(df, source)

    # Save report
    REPORT_FILE.write_text(report_md, encoding="utf-8")
    print(f"[Saved] Report → {REPORT_FILE}")
    print(f"[Saved] Report size: {REPORT_FILE.stat().st_size / 1024:.1f} KB")

    # Console summary
    print_console_summary(df)

    print(f"  ✓ Full report: {REPORT_FILE.resolve()}")
    print()


if __name__ == "__main__":
    main()
