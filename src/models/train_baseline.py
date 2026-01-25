from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import joblib

from src.utils.io import ensure_dir, save_json, read_csv, save_csv

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def time_split(df: pd.DataFrame, test_size: float = 0.25):
    df = df.sort_values("bucket")
    unique_buckets = sorted(pd.to_datetime(df["bucket"]).unique())
    cut = int(len(unique_buckets) * (1 - test_size))
    train_buckets = set(unique_buckets[:cut])
    test_buckets = set(unique_buckets[cut:])
    train_df = df[pd.to_datetime(df["bucket"]).isin(train_buckets)].copy()
    test_df = df[pd.to_datetime(df["bucket"]).isin(test_buckets)].copy()
    return train_df, test_df

def _get_feature_names(preprocessor) -> list[str]:
    names = []
    # TF-IDF
    tfidf = preprocessor.named_transformers_["tfidf"]
    tfidf_names = list(tfidf.get_feature_names_out())
    names.extend([f"tfidf:{n}" for n in tfidf_names])

    # Numeric
    num_names = list(preprocessor.transformers_[1][2])  # columns
    names.extend([f"num:{n}" for n in num_names])
    return names

def main(cfg_path: str = "configs/config.yaml") -> None:
    cfg = load_config(cfg_path)

    data_path = Path("data/processed/city_bucket_docs.csv")
    if not data_path.exists():
        raise FileNotFoundError("Build features first: python -m src.features.build_city_month_docs")

    df = read_csv(data_path)
    df["bucket"] = pd.to_datetime(df["bucket"])
    df = df.dropna(subset=["doc", "high_risk"])

    tfidf_max_features = int(cfg["features"]["tfidf_max_features"])
    ngram_range = tuple(cfg["features"]["tfidf_ngram_range"])

    train_df, test_df = time_split(df, test_size=0.25)

    X_train = train_df[["doc", "sent_mean", "n_articles"]]
    y_train = train_df["high_risk"].astype(int)

    X_test = test_df[["doc", "sent_mean", "n_articles"]]
    y_test = test_df["high_risk"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=tfidf_max_features, ngram_range=ngram_range), "doc"),
            ("num", Pipeline([("scaler", StandardScaler())]), ["sent_mean", "n_articles"]),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=int(cfg.get("seed", 42)),
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
    f1 = float(f1_score(y_test, pred)) if len(np.unique(y_test)) > 1 else float("nan")

    outputs = ensure_dir("outputs")

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "auc_roc": auc,
        "f1": f1,
        "class_report": classification_report(y_test, pred, output_dict=True),
        "note": "Label is synthetic (next-window cue_count). Replace with ACLED labels when available."
    }
    save_json(metrics, Path(outputs) / "metrics.json")

    # Feature importance (linear model coefficients)
    pre_fitted = pipe.named_steps["pre"]
    feat_names = _get_feature_names(pre_fitted)

    coefs = pipe.named_steps["clf"].coef_.ravel()
    fi = pd.DataFrame({"feature": feat_names, "coef": coefs})
    fi["abs_coef"] = fi["coef"].abs()
    fi = fi.sort_values("abs_coef", ascending=False)

    top = fi.head(30).reset_index(drop=True)
    save_csv(top, Path(outputs) / "feature_importance_top.csv")

    # Plot ROC curve
    plt.figure()
    if len(np.unique(y_test)) > 1:
        # manual ROC without sklearn display to keep deps minimal
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Baseline TF-IDF + Sentiment + Volume)")
    else:
        plt.text(0.1, 0.5, "ROC undefined (single-class test split)")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(Path(outputs) / "roc_curve.png", dpi=160)
    plt.close()

    # Plot top feature coefficients
    plt.figure(figsize=(10, 8))
    # show signed coefs but sort by abs
    plot_df = top.iloc[::-1]  # reverse for barh ascending
    plt.barh(plot_df["feature"], plot_df["coef"])
    plt.xlabel("Coefficient (log-odds)")
    plt.title("Top Features (Baseline)")
    plt.tight_layout()
    plt.savefig(Path(outputs) / "feature_importance_top.png", dpi=160)
    plt.close()

    # Save model
    joblib.dump(pipe, Path(outputs) / "baseline_model.joblib")

    print(f"Saved metrics -> {Path(outputs) / 'metrics.json'}")
    print(f"Saved ROC plot -> {Path(outputs) / 'roc_curve.png'}")
    print(f"Saved top features -> {Path(outputs) / 'feature_importance_top.csv'}")
    print(f"Saved model -> {Path(outputs) / 'baseline_model.joblib'}")
    print(f"AUC={auc:.3f} | F1={f1:.3f}")

if __name__ == "__main__":
    main()
