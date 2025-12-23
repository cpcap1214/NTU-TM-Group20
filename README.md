# Restaurant Review Text Mining

This project builds a full text mining pipeline for Google Maps restaurant reviews:
- Text cleaning, jieba tokenization, and stopword removal.
- Sentiment classification (TF-IDF + Naive Bayes / Logistic Regression).
- Topic modeling (LDA).
- Keyword analysis (TF-IDF and frequency).
- Output CSVs for analysis and reporting.

## Files
- `text_mining_project.py`: main pipeline script.
- `stopwords_zh.txt`: stopword list (edit for your domain).
- `final_reviews_for_analysis.csv`: input data.
- `outputs/`: generated outputs.

## Install
```
python3 -m pip install pandas numpy scikit-learn jieba
```

## Run
```
python3 text_mining_project.py --input final_reviews_for_analysis.csv --output outputs
```

Per-place outputs:
```
python3 text_mining_project.py --input final_reviews_for_analysis.csv --output outputs --per-place
```

## Output files
- `outputs/preprocessed_reviews.csv`
- `outputs/sentiment_metrics.json`
- `outputs/sentiment_confusion_matrix.csv`
- `outputs/sentiment_predictions.csv`
- `outputs/topic_terms.csv`
- `outputs/positive_top_keywords.csv`
- `outputs/negative_top_keywords.csv`
- `outputs/top_word_frequency.csv`
- `outputs/summary.json`
- `outputs/per_place/per_place_summary.csv`
- `outputs/per_place/<place_name>/summary.json`
- `outputs/per_place/<place_name>/positive_top_keywords.csv`
- `outputs/per_place/<place_name>/negative_top_keywords.csv`
- `outputs/per_place/<place_name>/topic_terms.csv` (if enough docs)

## Notes
- Sentiment labels are derived from star ratings: 4-5 positive, 1-2 negative, 3 neutral.
- Adjust `stopwords_zh.txt` to refine keyword and topic quality.
