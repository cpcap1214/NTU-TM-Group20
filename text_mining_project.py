#!/usr/bin/env python3
"""Text mining pipeline for Google Maps restaurant reviews."""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd


def _safe_import_jieba():
    try:
        import jieba  # type: ignore
    except Exception as exc:  # pragma: no cover - import check
        raise SystemExit(
            "Missing dependency: jieba. Install with: python3 -m pip install jieba"
        ) from exc
    return jieba


def load_user_dict(jieba_module, path: str) -> None:
    if not path or not os.path.exists(path):
        return
    jieba_module.load_userdict(path)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove punctuation and special symbols, keep CJK/letters/digits.
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", text)
    # Collapse repeated characters (e.g. 哈哈哈 -> 哈).
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_stopwords(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def tokenize(text: str, jieba_module, stopwords: set) -> str:
    if not text:
        return ""
    tokens = [t for t in jieba_module.cut(text) if t and t.strip()]
    filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(filtered)


def label_sentiment(rating: float) -> str:
    if pd.isna(rating):
        return "negative"
    if rating >= 4:
        return "positive"
    return "negative"


def compute_keywords(texts: pd.Series, top_n: int = 30):
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_docs = len(texts)
    if n_docs < 4:
        min_df = 1
        max_df = 1.0
    else:
        min_df = 3
        max_df = 0.9

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        max_df=max_df,
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=["term", "score"])

    scores = np.asarray(tfidf.mean(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:top_n]
    return pd.DataFrame({"term": terms[top_idx], "score": scores[top_idx]})


def compute_discriminative_keywords(
    pos_texts: pd.Series, neg_texts: pd.Series, top_k: int
):
    from sklearn.feature_extraction.text import TfidfVectorizer

    pos_texts = pos_texts.dropna()
    neg_texts = neg_texts.dropna()
    all_texts = pd.concat([pos_texts, neg_texts], ignore_index=True)

    n_docs = len(all_texts)
    if n_docs < 4:
        min_df = 1
        max_df = 1.0
    else:
        min_df = 3
        max_df = 0.9

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=min_df,
        max_df=max_df,
    )

    try:
        tfidf_all = vectorizer.fit_transform(all_texts)
    except ValueError:
        empty = pd.DataFrame(columns=["term", "score"])
        return empty, empty

    terms = np.array(vectorizer.get_feature_names_out())
    pos_n = len(pos_texts)
    neg_n = len(neg_texts)

    if pos_n == 0 or neg_n == 0:
        empty = pd.DataFrame(columns=["term", "score"])
        return empty, empty

    pos_mean = np.asarray(tfidf_all[:pos_n].mean(axis=0)).ravel()
    neg_mean = np.asarray(tfidf_all[pos_n : pos_n + neg_n].mean(axis=0)).ravel()
    diff = pos_mean - neg_mean

    pos_idx = np.argsort(diff)[::-1]
    neg_idx = np.argsort(diff)

    pos_terms = [(terms[i], diff[i]) for i in pos_idx if diff[i] > 0]
    neg_terms = [(terms[i], -diff[i]) for i in neg_idx if diff[i] < 0]

    pos_df = pd.DataFrame(pos_terms[:top_k], columns=["term", "score"])
    neg_df = pd.DataFrame(neg_terms[:top_k], columns=["term", "score"])
    return pos_df, neg_df


def topic_model(texts: pd.Series, n_topics: int, n_top_terms: int):
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=3,
        max_df=0.9,
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
    )
    doc_topics = lda.fit_transform(dtm)
    terms = np.array(vectorizer.get_feature_names_out())

    rows = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_idx = np.argsort(topic_weights)[::-1][:n_top_terms]
        rows.append(
            {
                "topic": f"topic_{topic_idx + 1}",
                "terms": " ".join(terms[top_idx]),
            }
        )
    topics_df = pd.DataFrame(rows)
    return lda, topics_df, doc_topics


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def per_place_analysis(
    df: pd.DataFrame,
    output_dir: str,
    n_topics: int,
    n_top_terms: int,
    top_keywords: int,
    min_docs_for_topics: int,
):
    ensure_dir(output_dir)
    rows = []
    grouped = df.groupby("place_name", dropna=False)
    for place, group in grouped:
        place_safe = re.sub(r"[^0-9A-Za-z\\u4e00-\\u9fff]+", "_", str(place))[:80]
        place_dir = os.path.join(output_dir, place_safe)
        ensure_dir(place_dir)

        summary = {
            "place_name": place,
            "rows": int(len(group)),
            "avg_rating": float(group["rating"].mean()),
            "sentiment_distribution": group["sentiment_label"].value_counts().to_dict(),
        }
        with open(os.path.join(place_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        pos = group[group["sentiment_label"] == "positive"]
        neg = group[group["sentiment_label"] == "negative"]
        if len(pos) > 0 and len(neg) > 0:
            pos_kw, neg_kw = compute_discriminative_keywords(
                pos["processed_text"], neg["processed_text"], top_keywords
            )
        else:
            pos_kw = compute_keywords(pos["processed_text"], top_keywords)
            neg_kw = compute_keywords(neg["processed_text"], top_keywords)

        if not pos_kw.empty:
            pos_kw.to_csv(os.path.join(place_dir, "positive_top_keywords.csv"), index=False)
        if not neg_kw.empty:
            neg_kw.to_csv(os.path.join(place_dir, "negative_top_keywords.csv"), index=False)

        if len(group) >= min_docs_for_topics:
            _, topics_df, _ = topic_model(group["processed_text"], n_topics, n_top_terms)
            topics_df.to_csv(os.path.join(place_dir, "topic_terms.csv"), index=False)

        rows.append(summary)

    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "per_place_summary.csv"), index=False)


def place_top_keywords(df: pd.DataFrame, place_name: str, top_k: int):
    target = df[df["place_name"] == place_name]
    if target.empty:
        raise SystemExit(f"Place not found: {place_name}")

    pos = target[target["sentiment_label"] == "positive"]
    neg = target[target["sentiment_label"] == "negative"]

    if len(pos) > 0 and len(neg) > 0:
        pos_kw, neg_kw = compute_discriminative_keywords(
            pos["processed_text"], neg["processed_text"], top_k
        )
    else:
        pos_kw = compute_keywords(pos["processed_text"], top_k) if len(pos) > 0 else pd.DataFrame()
        neg_kw = compute_keywords(neg["processed_text"], top_k) if len(neg) > 0 else pd.DataFrame()

    print("place_name:", place_name)
    print("top_positive:")
    if pos_kw.empty:
        print("  (no data)")
    else:
        for term in pos_kw["term"].head(top_k).tolist():
            print("  -", term)
    print("top_negative:")
    if neg_kw.empty:
        print("  (no data)")
    else:
        for term in neg_kw["term"].head(top_k).tolist():
            print("  -", term)


def sentiment_model(texts: pd.Series, labels: pd.Series, model_type: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=3,
        max_df=0.9,
    )

    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    if model_type == "nb":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=sorted(labels.unique()))

    return model, vectorizer, report, cm, (X_test, y_test)


def fit_sentiment_model(texts: pd.Series, labels: pd.Series, model_type: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=3,
        max_df=0.9,
    )
    X = vectorizer.fit_transform(texts)

    if model_type == "nb":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X, labels)
    return model, vectorizer


def main():
    parser = argparse.ArgumentParser(description="Restaurant review text mining")
    parser.add_argument("--input", default="final_reviews_for_analysis.csv")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--stopwords", default="stopwords_zh.txt")
    parser.add_argument("--user-dict", default="user_dict.txt")
    parser.add_argument("--model", choices=["nb", "logreg"], default="logreg")
    parser.add_argument("--topics", type=int, default=6)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--top-keywords", type=int, default=30)
    parser.add_argument("--per-place", action="store_true", help="Run per-place outputs")
    parser.add_argument("--min-place-docs", type=int, default=20)
    parser.add_argument("--place-name", type=str, default="")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    ensure_dir(args.output)

    df = pd.read_csv(args.input)
    if "review_text" not in df.columns or "rating" not in df.columns:
        raise SystemExit("Missing required columns: review_text, rating")

    jieba = _safe_import_jieba()
    stopwords = load_stopwords(args.stopwords)
    load_user_dict(jieba, args.user_dict)

    df["cleaned_text"] = df["review_text"].map(clean_text)
    df["processed_text"] = df["cleaned_text"].map(lambda x: tokenize(x, jieba, stopwords))
    df["sentiment_label"] = df["rating"].map(label_sentiment)

    preprocessed_path = os.path.join(args.output, "preprocessed_reviews.csv")
    df.to_csv(preprocessed_path, index=False)

    # Sentiment model
    sentiment_df = df[df["processed_text"].str.len() > 0].copy()
    model, vectorizer, report, cm, (X_test, y_test) = sentiment_model(
        sentiment_df["processed_text"], sentiment_df["sentiment_label"], args.model
    )
    preds = model.predict(X_test)

    metrics_path = os.path.join(args.output, "sentiment_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    cm_path = os.path.join(args.output, "sentiment_confusion_matrix.csv")
    cm_df = pd.DataFrame(
        cm,
        index=sorted(sentiment_df["sentiment_label"].unique()),
        columns=sorted(sentiment_df["sentiment_label"].unique()),
    )
    cm_df.to_csv(cm_path)

    # Predictions on full data
    full_tfidf = vectorizer.transform(sentiment_df["processed_text"])
    sentiment_df["predicted_sentiment"] = model.predict(full_tfidf)
    predictions_path = os.path.join(args.output, "sentiment_predictions.csv")
    sentiment_df.to_csv(predictions_path, index=False)

    # Topic modeling
    lda, topics_df, doc_topics = topic_model(
        sentiment_df["processed_text"], args.topics, args.top_terms
    )
    topics_path = os.path.join(args.output, "topic_terms.csv")
    topics_df.to_csv(topics_path, index=False)

    # Keyword analysis
    positive_df = sentiment_df[sentiment_df["sentiment_label"] == "positive"]
    negative_df = sentiment_df[sentiment_df["sentiment_label"] == "negative"]

    if len(positive_df) > 0:
        pos_keywords = compute_keywords(positive_df["processed_text"], args.top_keywords)
        pos_keywords.to_csv(os.path.join(args.output, "positive_top_keywords.csv"), index=False)

    if len(negative_df) > 0:
        neg_keywords = compute_keywords(negative_df["processed_text"], args.top_keywords)
        neg_keywords.to_csv(os.path.join(args.output, "negative_top_keywords.csv"), index=False)

    # Consistency analysis
    consistency = (
        sentiment_df["sentiment_label"] == sentiment_df["predicted_sentiment"]
    ).mean()

    summary = {
        "rows": int(len(df)),
        "non_empty_text": int(len(sentiment_df)),
        "label_distribution": sentiment_df["sentiment_label"].value_counts().to_dict(),
        "prediction_consistency": float(consistency),
        "output_dir": args.output,
    }

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Quick word frequency snapshot
    all_tokens = " ".join(sentiment_df["processed_text"]).split()
    freq = Counter(all_tokens).most_common(50)
    freq_df = pd.DataFrame(freq, columns=["term", "count"])
    freq_df.to_csv(os.path.join(args.output, "top_word_frequency.csv"), index=False)

    if args.per_place:
        per_place_analysis(
            sentiment_df,
            os.path.join(args.output, "per_place"),
            args.topics,
            args.top_terms,
            args.top_keywords,
            args.min_place_docs,
        )

    if args.place_name:
        place_top_keywords(sentiment_df, args.place_name, args.top_k)

    print("Done. Outputs in:", args.output)


if __name__ == "__main__":
    main()
