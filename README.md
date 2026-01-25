# City-Level Conflict Risk from News Text Signals

*One-sentence tagline:* A small end-to-end pipeline that turns GDELT news into city-month text signals and trains a simple baseline risk model.

## Abstract
This project is a student-built NLP prototype for turning news text into city-level, month-bucketed signals.
It downloads articles from GDELT, aggregates them into city-month documents, and builds TF-IDF plus numeric features.
A baseline classifier is trained to predict a forward-looking risk label.
The current label is synthetic (next-month conflict cue counts) so the pipeline can run without external event data.
The structure is meant to be swapped to ACLED labels once those data are added.
No experimental claims are made beyond a runnable baseline pipeline.

## What this project does
- Fetches news articles from GDELT and stores them in a local CSV.
- Buckets articles into city-month documents.
- Builds TF-IDF text features plus numeric cues (e.g., sentiment and keyword counts).
- Trains a baseline model and saves metrics and plots.
- Leaves a clear hook for replacing the synthetic label with ACLED event counts.

## Data
- **Current source:** GDELT news articles.
- **Planned source:** ACLED for conflict event labels (not yet integrated).
- **Labeling note:** the current label is **synthetic** (next-month conflict cue counts) and is only a stand-in for real event labels.

## Method overview
The pipeline builds city-month documents, extracts TF-IDF and numeric features, and trains a baseline classifier to predict a next-window risk label.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Fetch GDELT articles
python -m src.ingest.fetch_gdelt

# 2) Build city-month docs + features (and forward-looking label)
python -m src.features.build_city_month_docs

# 3) Train baseline model + save metrics/plots
python -m src.models.train_baseline
```

## Outputs
- `data/raw/gdelt_articles.csv`
- `data/processed/city_bucket_docs.csv`
- `outputs/metrics.json`
- `outputs/roc_curve.png`
- `outputs/feature_importance_top.csv`
- `outputs/feature_importance_top.png`

## Limitations
- The label is synthetic and should not be treated as real conflict outcomes.
- The baseline model is simple and not tuned for predictive performance.
- No temporal or geographic generalization checks are included yet.
- GDELT coverage and news bias are not corrected for in this version.

## Roadmap
1. Integrate ACLED labels for real conflict event outcomes.
2. Add strict location-based splits to test geographic generalization.
3. Run ablations (numeric-only vs text-only vs combined features).
4. Add Arabic modeling via AraBERT for Arabic-language coverage.
5. Add structured event extraction for more granular signals.

## Citation / disclaimer
This project is for research exploration only. Do not use it for operational or policy decisions. Correlation is not causation.

## Project context
[PASTE MY PROJECT DESCRIPTION + METHOD TEXT HERE]
