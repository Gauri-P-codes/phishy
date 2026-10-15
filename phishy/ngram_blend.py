"""
Does a character n-gram model improve recall over XGBoost?
Blends the two models' scores and compares RECALL AT FIXED PRECISION (the fair test).

  python ngram_blend.py --data data/PhiUSIIL_Phishing_URL_Dataset.csv --url-col URL --label-col label --invert-labels --n-per-class 50000
"""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score

import phishy  # noqa: F401
from phishy import data
from phishy.features import parse_url
from phishy.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--url-col", default="url")
ap.add_argument("--label-col", default="label")
ap.add_argument("--invert-labels", action="store_true")
ap.add_argument("--n-per-class", type=int, default=10000)
a = ap.parse_args()


def recall_at_precision(y, p, target):
    """Best recall achievable while keeping precision >= target."""
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


df = data.balance(data.load_url_csv(a.data, a.url_col, a.label_col, a.invert_labels))
df = df.groupby("label").head(a.n_per_class).reset_index(drop=True)

# 1) existing XGBoost model (same split as phishy.train)
bundle, _ = train(df)
train_df, test_df = data.group_split(df)
y = test_df["label"].values
p_xgb = bundle["model"].predict_proba(bundle["builder"].transform(test_df["url"]))[:, 1]

# 2) character n-gram model on the normalised URL (no scheme / www)
norm = lambda s: s.map(lambda u: parse_url(u).full)
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=5,
                        max_features=300_000, sublinear_tf=True, dtype=np.float32)
Xtr = tfidf.fit_transform(norm(train_df["url"]))
lr = LogisticRegression(max_iter=1000, C=3).fit(Xtr, train_df["label"])
p_lr = lr.predict_proba(tfidf.transform(norm(test_df["url"])))[:, 1]

# 3) compare, blending with different weights (w = weight on XGBoost)
print(f"{'model':<18}{'AUC':>8}{'recall@P90':>12}{'recall@P95':>12}")
for name, p in [("xgboost only", p_xgb), ("ngram only", p_lr),
                ("blend 70/30", 0.7 * p_xgb + 0.3 * p_lr),
                ("blend 50/50", 0.5 * p_xgb + 0.5 * p_lr),
                ("blend 30/70", 0.3 * p_xgb + 0.7 * p_lr)]:
    print(f"{name:<18}{roc_auc_score(y, p):>8.4f}"
          f"{recall_at_precision(y, p, 0.90):>12.4f}{recall_at_precision(y, p, 0.95):>12.4f}")
