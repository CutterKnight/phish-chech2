"""Week 5: train several models on several feature sets and compare them.

Run:  python train.py          (takes a few minutes)
Writes: models/metrics.json  and  models/<set name>.joblib  (logistic regression of each set)
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from feature_sets import FEATURE_SETS
from settings import RANDOM_STATE, TEST_SIZE
from urlfeatures import extract

DATA = Path("data/urls.csv")
OUT = Path("models")


def algorithms():
    return {
        "Baseline (always 'legitimate')": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }


def split_indexes(df, kind):
    """random: any URL can be in either part.  by_host: all URLs of one host stay together,
    so the test part contains hosts the model has never seen (a harder and fairer test)."""
    idx = list(range(len(df)))
    if kind == "random":
        train, test = train_test_split(idx, test_size=TEST_SIZE, stratify=df["is_phishing"], random_state=RANDOM_STATE)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train, test = next(gss.split(df, groups=df["host"]))
    return list(train), list(test)


def main():
    if not DATA.exists():
        raise SystemExit("data/urls.csv not found. Run prepare_data.py first.")
    df = pd.read_csv(DATA)
    print(f"Computing the features of {len(df)} URLs ...")
    X_all = pd.DataFrame([extract(u) for u in df["url"]])
    y = df["is_phishing"]
    OUT.mkdir(exist_ok=True)
    results, sets_used = [], {}

    for set_name, cols in FEATURE_SETS.items():
        cols = list(X_all.columns) if cols is None else list(cols)
        unknown = [c for c in cols if c not in X_all.columns]
        if unknown:
            raise SystemExit(f"Feature set {set_name}: unknown features: {unknown}")
        sets_used[set_name] = cols
        X = X_all[cols]
        for kind in ("random", "by_host"):
            tr, te = split_indexes(df, kind)
            print(f"\n=== {set_name} | split={kind} | {len(cols)} features | {len(tr)} train / {len(te)} test ===")
            for algo_name, estimator in algorithms().items():
                pipe = Pipeline([("scale", StandardScaler()), ("model", estimator)])
                pipe.fit(X.iloc[tr], y.iloc[tr])
                pred = pipe.predict(X.iloc[te])
                tn, fp, fn, tp = confusion_matrix(y.iloc[te], pred, labels=[0, 1]).ravel()
                row = {"split": kind, "set": set_name, "algorithm": algo_name,
                       "accuracy": accuracy_score(y.iloc[te], pred),
                       "precision": precision_score(y.iloc[te], pred, zero_division=0),
                       "recall": recall_score(y.iloc[te], pred, zero_division=0),
                       "f1": f1_score(y.iloc[te], pred, zero_division=0),
                       "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
                results.append(row)
                print(f"{algo_name:32s} accuracy={row['accuracy']:.3f} recall={row['recall']:.3f} "
                      f"precision={row['precision']:.3f} f1={row['f1']:.3f}")
                if algo_name == "Logistic Regression" and kind == "by_host":
                    joblib.dump(pipe, OUT / f"{set_name}.joblib")

    info = {"trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_urls": int(len(df)),
            "n_hosts": int(df["host"].nunique()), "test_size": TEST_SIZE, "random_state": RANDOM_STATE,
            "sets": sets_used, "results": results}
    (OUT / "metrics.json").write_text(json.dumps(info, indent=2))
    (OUT / "columns.json").write_text(json.dumps(sets_used, indent=2))
    print("\nSaved models/metrics.json and one .joblib file per feature set.")


if __name__ == "__main__":
    main()
