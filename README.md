# City-Level Conflict Risk from News Text Signals (GDELT MVP)

This repo is a minimal, runnable **NLP-first** prototype: fetches news articles from **GDELT**, aggregates them into **city-month documents**, builds text signals (TF–IDF + sentiment + cue counts), and trains a **baseline classifier** to predict *next-window* risk.

> Note: the label in this MVP is **synthetic** (next-month conflict cue counts) to validate the pipeline end-to-end. Swap the label with ACLED `Conflict_Events_Next` once you load event data.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Fetch GDELT articles
python -m src.ingest.fetch_gdelt

# 2) Build city-month docs + features (and forward-looking label)
python -m src.features.build_city_month_docs

# 3) Train baseline model + save metrics/plots
python -m src.models.train_baseline
```

Outputs are written to:
- `data/raw/gdelt_articles.csv`
- `data/processed/city_bucket_docs.csv`
- `outputs/metrics.json`
- `outputs/roc_curve.png`
- `outputs/feature_importance_top.csv`
- `outputs/feature_importance_top.png`

## What to do next (to make it research-grade)
- Replace synthetic label with **ACLED next-month conflict event counts**
- Add ablations: numeric-only vs text-only vs combined
- Add stronger model: transformer embeddings (BERT/AraBERT) pooled per city-window
- Add robustness checks: time split + location split

