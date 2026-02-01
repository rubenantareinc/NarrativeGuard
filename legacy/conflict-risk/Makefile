.PHONY: setup structured fetch features merge train_regression train_classification ablate_regression

setup:
	pip install -r requirements.txt

structured:
	python legacy/conflict-risk/src/ingest/load_structured.py

fetch:
	python legacy/conflict-risk/src/ingest/fetch_gdelt.py

features:
	python legacy/conflict-risk/src/features/build_city_month_docs.py

merge:
	python legacy/conflict-risk/src/features/merge_structured_text.py

train_regression:
	python legacy/conflict-risk/src/models/train_baseline.py --task regression --features combined

train_classification:
	python legacy/conflict-risk/src/models/train_baseline.py --task classification --features combined

ablate_regression:
	python legacy/conflict-risk/src/models/train_baseline.py --task regression --features structured_only
	python legacy/conflict-risk/src/models/train_baseline.py --task regression --features text_only
	python legacy/conflict-risk/src/models/train_baseline.py --task regression --features combined
