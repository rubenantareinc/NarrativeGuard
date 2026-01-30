# Data Documentation

This project uses narrative inconsistency datasets built from long-form, AI-generated narratives and curated contradiction annotations. The repository is structured to keep raw sources immutable and track processed artifacts separately.

## Directory layout

- `data/raw/`: Original source files (unmodified). Examples include synthetic narratives, source prompts, and raw annotation exports.
- `data/processed/`: Cleaned, sentence-segmented, and tokenized datasets used in experiments.
- `data/annotations/`: Human labels for contradiction spans, error categories, and narrative consistency scores.

## Dataset schema (processed)

| Field | Type | Description |
| --- | --- | --- |
| `doc_id` | string | Unique document identifier |
| `segment_id` | string | Sentence/segment identifier within document |
| `text` | string | Narrative segment |
| `label` | int | 1 = inconsistency detected, 0 = consistent |
| `source` | string | Dataset or prompt source |
| `split` | string | train/val/test |

## Annotation guidelines

Annotators flag contradictions that violate earlier narrative facts or logical coherence. Labels also capture error category (e.g., long-range dependency, implicit conflict) for error analysis in `src/analysis/error_analyzer.py`.

## Data use

This repository ships only metadata stubs and documentation. Place raw datasets under `data/raw/` and update `data/processed/` with derived artifacts.
