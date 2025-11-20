"""EDA pipeline for Financial News dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

RAW_DATA = Path("data/raw/raw_analyst_ratings.csv")
OUTPUT_DIR = Path("data/processed/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    stats = df[["headline_len_chars", "headline_len_words"]].describe()
    stats.to_csv(OUTPUT_DIR / "headline_length_stats.csv")


def publisher_analysis(df: pd.DataFrame) -> None:
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


def time_series_analysis(df: pd.DataFrame) -> None:
    daily_counts = (
        df.groupby("publish_date")
        .size()
        .rename("article_count")
        .reset_index()
        .sort_values("publish_date")
    )
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
    df = load_data()
    descriptive_stats(df)
    publisher_analysis(df)
    time_series_analysis(df)
    topic_modeling(df)
    print(
        f"EDA artifacts written to {OUTPUT_DIR} "
        f"(rows={len(df):,}, publishers={df['publisher'].nunique():,})"
    )


if __name__ == "__main__":
    main()

