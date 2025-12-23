#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Rating-based sentiment labeling")
    parser.add_argument("input_csv", nargs="?", default="comment.csv",
                        help="Input CSV file (default: reviews.csv)")
    parser.add_argument("output_csv", nargs="?", default="labeled_reviews.csv",
                        help="Output CSV file (default: labeled_reviews.csv)")
    args = parser.parse_args()

    # 讀取 CSV（保留原始文字，允許逗號與換行）
    df = pd.read_csv(args.input_csv)

    original_count = len(df)

    # 1) 將 rating 轉成數值，無法轉換者視為 invalid
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    invalid_numeric_mask = df["rating"].isna()
    invalid_numeric_count = invalid_numeric_mask.sum()
    df = df[~invalid_numeric_mask]

    # 2) rating 不在 1~5 範圍內視為 invalid
    out_of_range_mask = (df["rating"] < 1) | (df["rating"] > 5)
    out_of_range_count = out_of_range_mask.sum()
    df = df[~out_of_range_mask]

    # 3) place_name 或 review_text 為空值（NaN/空字串）移除
    # 先處理 NaN，再處理空字串（strip 後）
    place_empty = df["place_name"].isna() | (df["place_name"].astype(str).str.strip() == "")
    review_empty = df["review_text"].isna() | (df["review_text"].astype(str).str.strip() == "")
    empty_text_mask = place_empty | review_empty
    empty_text_count = empty_text_mask.sum()
    df = df[~empty_text_mask]

    # 情緒標註：不允許 neutral，rating >= 4 為 positive，其餘為 negative
    df["sentiment"] = df["rating"].apply(lambda x: "positive" if x >= 4 else "negative")

    # 輸出結果 CSV
    df.to_csv(args.output_csv, index=False)

    # 統計與輸出
    cleaned_count = len(df)
    sentiment_counts = df["sentiment"].value_counts()
    positive_count = sentiment_counts.get("positive", 0)
    negative_count = sentiment_counts.get("negative", 0)
    positive_ratio = positive_count / cleaned_count if cleaned_count else 0
    negative_ratio = negative_count / cleaned_count if cleaned_count else 0

    print(f"原始總筆數: {original_count}")
    print(f"清理後總筆數: {cleaned_count}")
    print(f"移除（rating 無法轉換）筆數: {invalid_numeric_count}")
    print(f"移除（rating 超出 1~5）筆數: {out_of_range_count}")
    print(f"移除（place_name/review_text 空值）筆數: {empty_text_count}")
    print(f"positive: {positive_count} ({positive_ratio:.2%})")
    print(f"negative: {negative_count} ({negative_ratio:.2%})")
    print("\n前5筆資料：")
    print(df.head(5))


if __name__ == "__main__":
    main()
