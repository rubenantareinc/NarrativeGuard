# Methods

## Goal and scope
The goal is to prototype a reproducible, city-level risk signal derived from news text, aggregated to monthly buckets, and to establish a transparent baseline model that can later be swapped to real event labels. The current scope is intentionally narrow: English-language GDELT news articles are collected for a fixed list of cities, transformed into city-month documents, and used to train a logistic regression classifier that predicts a forward-looking, synthetic “high risk” label for the next month. The pipeline is designed to be auditable and easy to replace with ACLED event labels once those data are integrated. No claims about real-world forecasting performance are made at this stage.【F:README.md†L1-L44】【F:legacy/conflict-risk/src/ingest/fetch_gdelt.py†L1-L78】【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L1-L87】

## Data sources and label definition (current synthetic; future ACLED next-month)
**Current source:** news articles are fetched from the GDELT 2.1 Document API using a city-specific query. Each query requires an explicit city mention plus a conflict-related query spine (e.g., “protest,” “attack,” “riot”), and the script collects article metadata and short text fields (title, description) for a fixed date range and city list defined in `legacy/conflict-risk/config.yaml`. The raw output is stored as `data/raw/gdelt_articles.csv`.【F:legacy/conflict-risk/src/ingest/fetch_gdelt.py†L1-L78】【F:legacy/conflict-risk/config.yaml†L1-L34】

**Current label (synthetic):** after aggregation to city-month documents, a forward-looking label is created by shifting the conflict cue count within each city by one month (i.e., the next month’s cue sum). This “cue_next” signal is binarized using a quantile threshold (default 0.75) to define “high_risk.” This label is explicitly synthetic and intended only for pipeline validation and baseline model training until event data are integrated.【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L41-L85】【F:legacy/conflict-risk/config.yaml†L35-L58】

**Future label (ACLED next-month):** the pipeline is structured to replace the synthetic label with ACLED-based next-month event counts per city-month. This will require joining city-month buckets to ACLED event counts and re-running the labeling step with real outcomes. The current implementation includes an explicit note in the model outputs indicating the synthetic nature of the label to prevent misuse.【F:README.md†L16-L44】【F:legacy/conflict-risk/src/models/train_baseline.py†L95-L106】

## Text preprocessing (language ID, cleanup, dedup strategy)
**Language ID:** the GDELT API returns a language field per article, but the current pipeline does not filter by language. For a methodologically sound next step, language filtering should be applied (e.g., restrict to English or add language-specific pipelines) using the provided `language` field before text aggregation. This is not yet implemented and should be treated as a known limitation in the current methods. 【F:legacy/conflict-risk/src/ingest/fetch_gdelt.py†L34-L47】【F:README.md†L53-L71】

**Cleanup:** titles and descriptions are lowercased, URLs are stripped, and repeated whitespace is collapsed. The processed fields are concatenated into a single document per article. This keeps preprocessing minimal and consistent with the baseline goal. 【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L12-L52】

**Deduplication:** no deduplication is currently applied. The GDELT API can return near-duplicate articles syndicated across domains, which can inflate cue counts and article volumes. A planned improvement is to implement URL-based and/or text-similarity deduplication prior to aggregation. This is flagged as a leakage/coverage risk in the methods narrative. 【F:legacy/conflict-risk/src/ingest/fetch_gdelt.py†L26-L47】【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L41-L70】

## Feature extraction
### TF-IDF
City-month documents are vectorized using TF-IDF with a configurable maximum feature count and n-gram range (default 5,000 features, 1–2 grams). The vectorizer is applied to the aggregated `doc` field. This provides a sparse bag-of-words representation suited for a linear baseline model. 【F:legacy/conflict-risk/config.yaml†L41-L48】【F:legacy/conflict-risk/src/models/train_baseline.py†L52-L75】

### Sentiment (VADER)
Each article’s concatenated text is scored with VADER’s compound sentiment score. For each city-month document, the mean sentiment score across its articles is computed as a numeric feature (`sent_mean`). This is included alongside TF-IDF in the baseline pipeline. 【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L39-L70】【F:legacy/conflict-risk/src/models/train_baseline.py†L63-L77】

### Conflict cue lexicon
A small, hand-authored conflict cue lexicon is defined in `legacy/conflict-risk/config.yaml` (e.g., “attack,” “clash,” “riot,” “airstrike”). Each article counts cue word occurrences using exact word-boundary matches, and the city-month document stores the sum of cue counts (`cue_sum`). Cue counts are used both for the synthetic label (next-month cue sum) and as a diagnostic signal during analysis. The current baseline model uses `sent_mean` and `n_articles` as numeric features; cue counts are not included as a direct feature to avoid tautological leakage into the synthetic label. If ACLED labels are introduced, cue counts can be re-evaluated for inclusion in the feature set. 【F:legacy/conflict-risk/config.yaml†L51-L58】【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L19-L85】【F:legacy/conflict-risk/src/models/train_baseline.py†L63-L77】

## Aggregation: article → city-time bucket (month); missingness indicators
Articles are bucketed into monthly periods using the configured frequency (`freq: "M"`). For each city-month group, all article texts are concatenated into a single document (`doc`) and several numeric summaries are computed: number of articles (`n_articles`), mean sentiment (`sent_mean`), and total cue count (`cue_sum`). Documents below a minimum length threshold are dropped to avoid ultra-sparse text representations. The current pipeline does not explicitly add missingness indicators for absent months or zero-article months; this is a planned extension. A practical implementation would add binary flags for missing buckets per city and an imputation strategy for numeric features if sparse months are retained. 【F:legacy/conflict-risk/config.yaml†L35-L48】【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L32-L83】

## Train/test split: time-based; note leakage risks
The dataset is split into training and test sets by time: city-month buckets are sorted chronologically, and the last 25% of months become the test set. This split prevents training on future time periods. However, it does not prevent leakage across cities (e.g., the same city appearing in both train and test) and does not address potential duplication across sources. These risks must be documented and mitigated in future versions (e.g., city-held-out splits, deduplication, or stricter time windows). 【F:legacy/conflict-risk/src/models/train_baseline.py†L17-L59】

## Model: logistic regression baseline (balanced classes)
The baseline model is a logistic regression classifier with `class_weight="balanced"` and a `liblinear` solver. TF-IDF features are combined with standardized numeric features (sentiment mean and article volume) using a `ColumnTransformer`. The model is intended to be transparent, fast to train, and easy to inspect for feature importance. The goal is to establish a reproducible baseline rather than maximize predictive performance. 【F:legacy/conflict-risk/src/models/train_baseline.py†L52-L116】

## Evaluation: AUROC, F1, thresholding; what “good” means here
Evaluation includes AUROC and F1 on the time-based test split. A default threshold of 0.5 is used to convert predicted probabilities into binary labels for F1 and classification reports. Given the synthetic label and the limited dataset size, “good” performance is defined narrowly: the model should beat random guessing (AUROC > 0.5) and exhibit stable behavior across months, without overfitting obvious keyword leakage. The primary goal is diagnostic: validate that the pipeline is wired correctly and produces interpretable metrics rather than to claim real-world predictive value. 【F:legacy/conflict-risk/src/models/train_baseline.py†L78-L107】【F:README.md†L44-L71】

## Error analysis plan: top FP/FN by city; news volume bias; geocoding issues (planned)
Planned error analysis will focus on:
1. **Top false positives/negatives by city-month:** review the highest-probability errors and examine their source articles and cue patterns.
2. **News volume bias:** compare prediction errors across buckets with low `n_articles` to check whether sparse news coverage inflates risk estimates.
3. **Geocoding issues (planned):** GDELT matches on city names and may include homonyms or mismatches; future work should log article locations, add geocoding validation, and track misattribution by city.

These analyses are not yet implemented in code but are straightforward additions using the existing outputs (`city_bucket_docs.csv` and `metrics.json`).【F:legacy/conflict-risk/src/features/build_city_month_docs.py†L60-L87】【F:legacy/conflict-risk/src/models/train_baseline.py†L95-L152】

## Ethics and responsible use
This project is an academic prototype and explicitly does not claim operational predictive capability. It is built on news data with known coverage and framing biases, and the current label is synthetic. Outputs should not be used for policy, security, or humanitarian decision-making. Once ACLED labels are integrated, additional safeguards should include bias audits, transparency on geographic coverage, and careful communication about uncertainty and limitations. Any deployment should follow responsible research practices and include human oversight. 【F:README.md†L72-L88】【F:legacy/conflict-risk/src/models/train_baseline.py†L95-L106】
