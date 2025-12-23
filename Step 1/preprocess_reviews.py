#!/usr/bin/env python3
import argparse
import jieba
import pandas as pd

def tokenize_only(text, min_token_length):
    tokens = jieba.lcut(text)
    cleaned = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) < min_token_length:
            continue
        cleaned.append(tok)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Chinese text preprocessing for reviews")
    parser.add_argument("input_csv", nargs="?", default="labeled_reviews.csv",
                        help="Input CSV file (default: labeled_reviews.csv)")
    parser.add_argument("output_csv", nargs="?", default="preprocessed_reviews.csv",
                        help="Output CSV file (default: preprocessed_reviews.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    original_count = len(df)

    def process_review(text):
        if pd.isna(text):
            text = ""
        text = str(text)
        tokens = tokenize_only(text, min_token_length=2)
        return " ".join(tokens)

    # 只處理 review_text，其餘欄位完整保留
    df["clean_text"] = df["review_text"].apply(process_review)

    df = df[["place_name", "rating", "review_text", "sentiment", "clean_text"]]
    df.to_csv(args.output_csv, index=False)

    print(f"處理前總筆數: {original_count}")
    print(f"處理後總筆數: {len(df)}")
    print("\n前5筆資料：")
    print(df.head(5))

    print("\n隨機抽樣 3 筆 clean_text：")
    sample = df["clean_text"].sample(n=min(3, len(df)), random_state=42)
    for i, text in enumerate(sample.tolist(), start=1):
        print(f"{i}: {text}")


if __name__ == "__main__":
    main()
