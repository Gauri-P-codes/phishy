"""
Train + evaluate + save.

  python -m phishy.train --data data/urls.csv --url-col url --label-col label
  python -m phishy.train --phishtank data/PhishTank.csv --tranco data/tranco.csv
  add  --ood-data data/other.csv  to test on a DIFFERENT dataset (the honest number)
"""
import argparse
import json

import joblib
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

import phishy  # noqa: F401  (registers features + models)
from config import DEFAULT_THRESHOLD, MODEL_PATH
from phishy import data
from phishy.features import FeatureBuilder
from phishy.models import MODELS


def metrics(y_true, prob, threshold=DEFAULT_THRESHOLD) -> dict:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, pred), 4),
        "precision_phishing": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall_phishing": round(recall_score(y_true, pred, zero_division=0), 4),
        "specificity_legitimate": round(tn / max(tn + fp, 1), 4),
        "f1_phishing": round(f1_score(y_true, pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
    }


def train(df, model_name="xgboost", include_network=False, threshold=DEFAULT_THRESHOLD):
    """Callable from a notebook too. Returns (bundle, report)."""
    train_df, test_df = data.group_split(df)

    builder = FeatureBuilder(include_network=include_network)
    X_train = builder.fit_transform(train_df["url"])
    X_test = builder.transform(test_df["url"])

    model = MODELS[model_name]()
    model.fit(X_train, train_df["label"])

    report = {
        "model": model_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train": metrics(train_df["label"], model.predict_proba(X_train)[:, 1], threshold),
        "test": metrics(test_df["label"], model.predict_proba(X_test)[:, 1], threshold),
    }
    bundle = {"builder": builder, "model": model, "threshold": threshold,
              "model_name": model_name, "report": report}
    return bundle, report


def evaluate_external(bundle, df):
    """Score a trained bundle on a completely different dataset."""
    X = bundle["builder"].transform(df["url"])
    prob = bundle["model"].predict_proba(X)[:, 1]
    return metrics(df["label"], prob, bundle["threshold"])


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV with url + label columns (both classes real URLs)")
    src.add_argument("--phishtank", help="PhishTank CSV (use with --tranco)")
    ap.add_argument("--tranco")
    ap.add_argument("--url-col", default="url")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--invert-labels", action="store_true",
                    help="use when the dataset's 1 means LEGITIMATE (PhiUSIIL)")
    ap.add_argument("--ood-invert-labels", action="store_true",
                    help="same, for --ood-data (its convention may differ)")
    ap.add_argument("--ood-data", help="second dataset for out-of-distribution evaluation")
    ap.add_argument("--model", default="xgboost", choices=list(MODELS))
    ap.add_argument("--n-per-class", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--network-features", action="store_true")
    ap.add_argument("--out", default=str(MODEL_PATH))
    a = ap.parse_args()

    if a.data:
        df = data.balance(data.load_url_csv(a.data, a.url_col, a.label_col, a.invert_labels))
        df = df.groupby("label").head(a.n_per_class).reset_index(drop=True)
    else:
        if not a.tranco:
            ap.error("--phishtank needs --tranco")
        df = data.build_phishtank_tranco(a.phishtank, a.tranco, a.n_per_class)
    print(f"Dataset: {len(df)} rows | label mix: {df['label'].value_counts().to_dict()}")

    bundle, report = train(df, a.model, a.network_features, a.threshold)
    if a.ood_data:
        ood = data.load_url_csv(a.ood_data, a.url_col, a.label_col, a.ood_invert_labels)
        report["out_of_distribution"] = evaluate_external(bundle, ood)

    print(json.dumps(report, indent=2))
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, a.out)
    print("Saved ->", a.out)


if __name__ == "__main__":
    main()
