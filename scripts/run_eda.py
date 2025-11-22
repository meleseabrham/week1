"""EDA pipeline for Financial News dataset.

This script performs comprehensive Exploratory Data Analysis including:
- Descriptive statistics and distribution analysis
- Publisher analysis with domain extraction
- Time series analysis (daily, hourly, weekday patterns)
- Topic modeling using LDA
- Statistical visualizations and evidence-based insights
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

matplotlib.use("Agg")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

RAW_DATA = Path("data/raw/raw_analyst_ratings.csv")
OUTPUT_DIR = Path("data/processed/eda")
FIG_DIR = Path("reports/figures/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_DATA,
        encoding_errors="replace",
        on_bad_lines="skip",
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"], utc=True, format="mixed")
    df["headline"] = df["headline"].fillna("")
    df["headline_len_chars"] = df["headline"].str.len()
    df["headline_len_words"] = df["headline"].str.count(r"\b\w+\b")
    df["publisher"] = df["publisher"].fillna("Unknown")
    df["publisher_domain"] = (
        df["publisher"].str.extract(r"@(.+)$")[0].str.lower().fillna("not_email")
    )
    df["publish_date"] = df["date"].dt.date
    df["publish_hour_utc"] = df["date"].dt.hour
    df["publish_dayofweek"] = df["date"].dt.day_name()
    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    """Compute descriptive statistics and distribution analysis for headline lengths."""
    stats_df = df[["headline_len_chars", "headline_len_words"]].describe()
    stats_df.to_csv(OUTPUT_DIR / "headline_length_stats.csv")
    
    # Statistical distribution analysis
    char_lengths = df["headline_len_chars"].dropna()
    word_lengths = df["headline_len_words"].dropna()
    
    # Test for normal distribution
    _, p_char_norm = stats.normaltest(char_lengths.sample(min(5000, len(char_lengths))))
    _, p_word_norm = stats.normaltest(word_lengths.sample(min(5000, len(word_lengths))))
    
    # Compute additional statistics
    additional_stats = {
        "char_length": {
            "mean": char_lengths.mean(),
            "median": char_lengths.median(),
            "std": char_lengths.std(),
            "skewness": stats.skew(char_lengths),
            "kurtosis": stats.kurtosis(char_lengths),
            "is_normal": p_char_norm > 0.05,
            "p_value_normality": float(p_char_norm),
        },
        "word_length": {
            "mean": word_lengths.mean(),
            "median": word_lengths.median(),
            "std": word_lengths.std(),
            "skewness": stats.skew(word_lengths),
            "kurtosis": stats.kurtosis(word_lengths),
            "is_normal": p_word_norm > 0.05,
            "p_value_normality": float(p_word_norm),
        },
    }
    
    # Save statistical analysis
    with open(OUTPUT_DIR / "statistical_analysis.json", "w") as f:
        json.dump(additional_stats, f, indent=2)
    
    # Create distribution visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Character length distribution
    axes[0, 0].hist(char_lengths, bins=50, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(char_lengths.mean(), color="r", linestyle="--", label=f"Mean: {char_lengths.mean():.1f}")
    axes[0, 0].axvline(char_lengths.median(), color="g", linestyle="--", label=f"Median: {char_lengths.median():.1f}")
    axes[0, 0].set_xlabel("Headline Length (Characters)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Headline Character Lengths")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Word length distribution
    axes[0, 1].hist(word_lengths, bins=30, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].axvline(word_lengths.mean(), color="r", linestyle="--", label=f"Mean: {word_lengths.mean():.1f}")
    axes[0, 1].axvline(word_lengths.median(), color="g", linestyle="--", label=f"Median: {word_lengths.median():.1f}")
    axes[0, 1].set_xlabel("Headline Length (Words)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Headline Word Counts")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plots for normality testing
    stats.probplot(char_lengths.sample(min(5000, len(char_lengths))), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot: Character Length vs Normal Distribution")
    axes[1, 0].grid(True, alpha=0.3)
    
    stats.probplot(word_lengths.sample(min(5000, len(word_lengths))), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot: Word Count vs Normal Distribution")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "headline_length_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Descriptive statistics computed (skewness: {additional_stats['char_length']['skewness']:.2f})")


def publisher_analysis(df: pd.DataFrame) -> None:
    """Analyze publisher distribution and create visualizations."""
    publisher_counts = (
        df.groupby("publisher")
        .size()
        .sort_values(ascending=False)
        .rename("article_count")
    )
    publisher_counts.to_csv(OUTPUT_DIR / "publisher_article_counts.csv")

    domain_counts = (
        df.groupby("publisher_domain")
        .size()
        .sort_values(ascending=False)
        .rename("article_count")
    )
    domain_counts.to_csv(OUTPUT_DIR / "publisher_domain_counts.csv")
    
    # Statistical analysis: Concentration metrics
    total_articles = publisher_counts.sum()
    top_10_pct = (publisher_counts.head(10).sum() / total_articles) * 100
    gini_coefficient = _calculate_gini(publisher_counts.values)
    
    concentration_stats = {
        "total_publishers": len(publisher_counts),
        "total_articles": int(total_articles),
        "top_10_percentage": float(top_10_pct),
        "gini_coefficient": float(gini_coefficient),
        "concentration_interpretation": "Highly concentrated" if gini_coefficient > 0.7 else "Moderately concentrated"
    }
    
    with open(OUTPUT_DIR / "publisher_concentration_stats.json", "w") as f:
        json.dump(concentration_stats, f, indent=2)
    
    # Visualizations
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top 20 publishers bar chart
    top_20 = publisher_counts.head(20)
    axes[0].barh(range(len(top_20)), top_20.values, color="steelblue")
    axes[0].set_yticks(range(len(top_20)))
    axes[0].set_yticklabels(top_20.index, fontsize=9)
    axes[0].set_xlabel("Number of Articles")
    axes[0].set_title(f"Top 20 Publishers by Article Count (Top 10 = {top_10_pct:.1f}% of total)")
    axes[0].grid(True, alpha=0.3, axis="x")
    axes[0].invert_yaxis()
    
    # Publisher distribution (log scale)
    axes[1].hist(publisher_counts.values, bins=50, edgecolor="black", alpha=0.7, color="coral")
    axes[1].set_xlabel("Articles per Publisher")
    axes[1].set_ylabel("Number of Publishers (Frequency)")
    axes[1].set_title("Distribution of Articles per Publisher (Power Law Distribution)")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "publisher_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Publisher analysis complete (Gini: {gini_coefficient:.3f}, Top 10: {top_10_pct:.1f}%)")


def _calculate_gini(values: np.ndarray) -> float:
    """Calculate Gini coefficient for concentration measurement."""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


def time_series_analysis(df: pd.DataFrame) -> None:
    """Perform time series analysis with statistical tests and visualizations."""
    daily_counts = (
        df.groupby("publish_date")
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values("publish_date")
    )
    daily_counts["publish_date"] = pd.to_datetime(daily_counts["publish_date"])
    daily_counts.to_csv(OUTPUT_DIR / "daily_publication_counts.csv", index=False)

    dow_counts = (
        df.groupby("publish_dayofweek")
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values("article_count", ascending=False)
    )
    dow_counts.to_csv(OUTPUT_DIR / "weekday_publication_counts.csv", index=False)

    hour_counts = (
        df.groupby("publish_hour_utc")
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values("publish_hour_utc")
    )
    hour_counts.to_csv(OUTPUT_DIR / "hourly_publication_counts.csv", index=False)
    
    # Statistical analysis: Test for weekday patterns
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_ordered = df["publish_dayofweek"].value_counts().reindex(weekday_order, fill_value=0)
    
    # Chi-square test for uniform distribution across weekdays
    expected = len(df) / 7
    chi2_stat, p_value = stats.chisquare(dow_ordered.values, f_exp=[expected] * 7)
    
    time_stats = {
        "date_range": {
            "start": str(daily_counts["publish_date"].min().date()),
            "end": str(daily_counts["publish_date"].max().date()),
            "total_days": int((daily_counts["publish_date"].max() - daily_counts["publish_date"].min()).days)
        },
        "weekday_analysis": {
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "is_uniform": p_value > 0.05,
            "interpretation": "Significant weekday pattern detected" if p_value < 0.05 else "No significant weekday pattern"
        },
        "peak_weekday": dow_counts.iloc[0].to_dict(),
        "peak_hour": int(hour_counts.loc[hour_counts["article_count"].idxmax(), "publish_hour_utc"])
    }
    
    with open(OUTPUT_DIR / "time_series_statistics.json", "w") as f:
        json.dump(time_stats, f, indent=2)
    
    # Visualizations
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Daily time series
    axes[0].plot(daily_counts["publish_date"], daily_counts["article_count"], linewidth=0.5, alpha=0.7)
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Articles per Day")
    axes[0].set_title("Daily Publication Volume Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)
    
    # Weekday distribution
    dow_plot = dow_counts.set_index("publish_dayofweek").reindex(weekday_order, fill_value=0)
    axes[1].bar(range(len(dow_plot)), dow_plot["article_count"], color="steelblue", edgecolor="black")
    axes[1].set_xticks(range(len(dow_plot)))
    axes[1].set_xticklabels(dow_plot.index, rotation=45, ha="right")
    axes[1].set_ylabel("Article Count")
    axes[1].set_title(f"Weekday Publication Pattern (χ²={chi2_stat:.1f}, p={p_value:.2e})")
    axes[1].axhline(expected, color="r", linestyle="--", label=f"Expected (uniform): {expected:.0f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Hourly distribution
    axes[2].bar(hour_counts["publish_hour_utc"], hour_counts["article_count"], color="coral", edgecolor="black")
    axes[2].set_xlabel("Hour of Day (UTC)")
    axes[2].set_ylabel("Article Count")
    axes[2].set_title("Hourly Publication Distribution (UTC)")
    axes[2].set_xticks(range(0, 24, 2))
    axes[2].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_series_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Time series analysis complete (Weekday pattern: p={p_value:.2e})")


def topic_modeling(df: pd.DataFrame, n_topics: int = 6, top_n: int = 10) -> None:
    vectorizer = CountVectorizer(
        stop_words="english", max_df=0.7, min_df=25, ngram_range=(1, 2)
    )
    dtm = vectorizer.fit_transform(df["headline"].fillna(""))
    lda = LatentDirichletAllocation(
        n_components=n_topics, learning_method="batch", random_state=42, n_jobs=-1
    )
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()

    topics: list[dict[str, str | int]] = []
    for idx, topic in enumerate(lda.components_, start=1):
        top_indices = topic.argsort()[-top_n:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        topics.append({"topic": idx, "keywords": keywords})

    (OUTPUT_DIR / "topic_keywords.json").write_text(
        json.dumps(topics, indent=2), encoding="utf-8"
    )


def main() -> None:
    """Main EDA pipeline execution."""
    print("=" * 60)
    print("Nova Financial Insights - Exploratory Data Analysis")
    print("=" * 60)
    
    print("\n[1/4] Loading and preprocessing data...")
    df = load_data()
    print(f"   Loaded {len(df):,} articles from {df['publisher'].nunique():,} publishers")
    
    print("\n[2/4] Computing descriptive statistics and distributions...")
    descriptive_stats(df)
    
    print("\n[3/4] Analyzing publisher patterns...")
    publisher_analysis(df)
    
    print("\n[4/4] Performing time series analysis...")
    time_series_analysis(df)
    
    print("\n[5/5] Running topic modeling (LDA)...")
    topic_modeling(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"✓ Artifacts written to: {OUTPUT_DIR}")
    print(f"✓ Visualizations saved to: {FIG_DIR}")
    print(f"✓ Dataset: {len(df):,} rows, {df['publisher'].nunique():,} publishers")
    print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

