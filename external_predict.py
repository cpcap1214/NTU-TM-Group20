#!/usr/bin/env python3
"""Predict sentiment on external restaurant reviews and output top keywords per place."""

import argparse
import os
import re

import pandas as pd

from text_mining_project import (
    clean_text,
    compute_keywords,
    compute_discriminative_keywords,
    ensure_dir,
    fit_sentiment_model,
    load_stopwords,
    load_user_dict,
    tokenize,
    _safe_import_jieba,
)


def main():
    parser = argparse.ArgumentParser(description="External dataset prediction")
    parser.add_argument("--train", default="final_reviews_for_analysis.csv")
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="outputs/external_per_place")
    parser.add_argument("--stopwords", default="stopwords_zh.txt")
    parser.add_argument("--user-dict", default="user_dict.txt")
    parser.add_argument("--model", choices=["nb", "logreg"], default="logreg")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    if "review_text" not in train_df.columns or "rating" not in train_df.columns:
        raise SystemExit("Training data missing required columns: review_text, rating")

    jieba = _safe_import_jieba()
    stopwords = load_stopwords(args.stopwords)
    load_user_dict(jieba, args.user_dict)

    train_df["cleaned_text"] = train_df["review_text"].map(clean_text)
    train_df["processed_text"] = train_df["cleaned_text"].map(
        lambda x: tokenize(x, jieba, stopwords)
    )
    train_df["sentiment_label"] = train_df["rating"].map(
        lambda x: "positive" if x > 4 else "negative"
    )
    train_df = train_df[train_df["processed_text"].str.len() > 0].copy()

    model, vectorizer = fit_sentiment_model(
        train_df["processed_text"], train_df["sentiment_label"], args.model
    )

    test_df = pd.read_csv(args.test)
    if "review_text" not in test_df.columns or "place_name" not in test_df.columns:
        raise SystemExit("Test data missing required columns: place_name, review_text")

    test_df["cleaned_text"] = test_df["review_text"].map(clean_text)
    test_df["processed_text"] = test_df["cleaned_text"].map(
        lambda x: tokenize(x, jieba, stopwords)
    )
    test_df = test_df[test_df["processed_text"].str.len() > 0].copy()

    test_tfidf = vectorizer.transform(test_df["processed_text"])
    test_df["predicted_sentiment"] = model.predict(test_tfidf)

    ensure_dir(args.output)
    test_df.to_csv(os.path.join(args.output, "external_predictions.csv"), index=False)

    grouped = test_df.groupby("place_name", dropna=False)
    for place, group in grouped:
        place_safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(place))[:80]
        place_dir = os.path.join(args.output, place_safe)
        ensure_dir(place_dir)

        pos = group[group["predicted_sentiment"] == "positive"]
        neg = group[group["predicted_sentiment"] == "negative"]

        if len(pos) > 0 and len(neg) > 0:
            pos_kw, neg_kw = compute_discriminative_keywords(
                pos["processed_text"], neg["processed_text"], args.top_k
            )
        else:
            pos_kw = compute_keywords(pos["processed_text"], args.top_k) if len(pos) > 0 else pd.DataFrame()
            neg_kw = compute_keywords(neg["processed_text"], args.top_k) if len(neg) > 0 else pd.DataFrame()

        if not pos_kw.empty:
            pos_kw.to_csv(os.path.join(place_dir, "positive_top_keywords.csv"), index=False)
        if not neg_kw.empty:
            neg_kw.to_csv(os.path.join(place_dir, "negative_top_keywords.csv"), index=False)

        print("place_name:", place)
        print("top_positive:")
        if pos_kw.empty:
            print("  (no data)")
        else:
            for term in pos_kw["term"].head(args.top_k).tolist():
                print("  -", term)
        print("top_negative:")
        if neg_kw.empty:
            print("  (no data)")
        else:
            for term in neg_kw["term"].head(args.top_k).tolist():
                print("  -", term)
        print("")


if __name__ == "__main__":
    main()
