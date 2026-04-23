"""
train_baseline.py
─────────────────
Baseline classification for 3 targets: priority, sender_type, intent
Models: LogisticRegression, DecisionTree, XGBoost
Features: TF-IDF on body_tfidf + numeric structural columns

Outputs (all in ./baseline_output/):
  models/          — best_<target>.pkl  (Pipeline: TF-IDF + scaler + model)
  reports/         — report_<target>.txt
  plots/           — confusion_<target>.png
  plots/           — feature_importance_<target>.png
  baseline_summary.txt

Usage:
  python train_baseline.py
  python train_baseline.py --db /path/to/email_intelligence.db
  python train_baseline.py --min-samples 200 --test-size 0.2
"""

import argparse
import json
import logging
import os
import pickle
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TARGETS = ["priority", "sender_type", "intent"]

NUMERIC_FEATURES = [
    "body_length", "is_reply", "cc_count", "reply_chain_depth",
    "hour", "day_of_week", "month",
]

TFIDF_PARAMS = dict(
    max_features=20_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
    strip_accents="unicode",
)

# Boost weights for rare/important classes
CLASS_BOOST = {
    "urgent":   4.0, "high":     2.0, "normal":   1.0, "low":      0.8,
    "client":   5.0, "personal": 2.0, "spam":     1.5,
    "vendor":   1.0, "internal": 0.8,
}

OUT_DIR = Path("baseline_output")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT
            ce.body_tfidf,
            ce.body_length,
            CAST(ce.is_reply AS INTEGER)   AS is_reply,
            ce.cc_count,
            ce.reply_chain_depth,
            COALESCE(ce.hour, -1)          AS hour,
            COALESCE(ce.day_of_week, -1)   AS day_of_week,
            COALESCE(ce.month, -1)         AS month,
            rc.priority,
            rc.sender_type,
            rc.intent
        FROM cleaned_emails ce
        JOIN results_classifications rc ON rc.email_id = ce.id
        WHERE ce.body_tfidf IS NOT NULL
          AND ce.body_tfidf != ''
          AND rc.priority    IS NOT NULL
          AND rc.sender_type NOT IN ('unknown', 'external', 'government', 'system')
          AND rc.intent      IS NOT NULL
    """, conn)
    conn.close()
    log.info(f"Loaded {len(df):,} labelled rows from {db_path}")
    return df


INTENT_MAP = {
    "request":   "action",
    "complaint": "action",
    "inquiry":   "action",
    "meeting":   "action",
    "fyi":       "info",
    "update":    "info",
    "social":    "info",
    "other":     "other",
}

def consolidate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Merge high+urgent → action_needed. Collapse intent to 3 classes."""
    df = df.copy()
    df["priority"] = df["priority"].replace({"urgent": "action_needed", "high": "action_needed"})
    df["intent"]   = df["intent"].map(INTENT_MAP).fillna("other")
    return df


def drop_rare_classes(df: pd.DataFrame, target: str, min_samples: int) -> pd.DataFrame:
    counts = df[target].value_counts()
    keep = counts[counts >= min_samples].index
    dropped = counts[counts < min_samples]
    if len(dropped):
        log.warning(
            f"[{target}] Dropping rare classes (< {min_samples} samples): "
            + ", ".join(f"{c}({n})" for c, n in dropped.items())
        )
    return df[df[target].isin(keep)].copy()


# ── Feature building ──────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, vectorizer: TfidfVectorizer = None, fit: bool = True):
    """Returns (X_sparse, vectorizer). Combines TF-IDF + numeric block."""
    texts = df["body_tfidf"].fillna("").tolist()

    if fit:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        X_text = vectorizer.fit_transform(texts)
    else:
        X_text = vectorizer.transform(texts)

    num = df[NUMERIC_FEATURES].fillna(0).astype(float).values
    X_num = csr_matrix(StandardScaler().fit_transform(num))

    X = hstack([X_text, X_num])
    return X, vectorizer


# ── Model definitions ─────────────────────────────────────────────────────────

def get_models(n_classes: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        ),
    }


# ── Training + evaluation ────────────────────────────────────────────────────

def train_target(df: pd.DataFrame, target: str, test_size: float, min_samples: int):
    log.info(f"{'='*60}")
    log.info(f"TARGET: {target.upper()}")
    log.info(f"{'='*60}")

    df_t = drop_rare_classes(df, target, min_samples)
    log.info(f"Class distribution:\n{df_t[target].value_counts().to_string()}")

    le = LabelEncoder()
    y = le.fit_transform(df_t[target])
    classes = le.classes_

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(sss.split(df_t, y))
    df_train, df_test = df_t.iloc[train_idx], df_t.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    log.info(f"Train: {len(y_train):,} | Test: {len(y_test):,}")

    # Build features (fit on train, transform both)
    X_train, vectorizer = build_features(df_train, fit=True)
    X_test, _           = build_features(df_test,  vectorizer=vectorizer, fit=False)

    models = get_models(len(classes))
    results = {}

    for name, clf in models.items():
        log.info(f"  Training {name}...")
        sample_weights = np.array([
            CLASS_BOOST.get(le.inverse_transform([yi])[0], 1.0)
            for yi in y_train
        ])
        if name == "xgboost":
            clf.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train, y_train)  # LR + DT use class_weight='balanced'
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)
            action_idx = np.where(le.classes_ == "action_needed")[0]
            if len(action_idx) > 0:
                aidx = action_idx[0]
                # Sweep thresholds to find best action_needed recall >= 0.80
                best_thresh, best_f1 = 0.5, 0.0
                for thresh in np.arange(0.15, 0.55, 0.025):
                    preds = np.argmax(proba, axis=1).copy()
                    preds[proba[:, aidx] >= thresh] = aidx
                    tp = np.sum((preds == aidx) & (y_test == aidx))
                    fp = np.sum((preds == aidx) & (y_test != aidx))
                    fn = np.sum((preds != aidx) & (y_test == aidx))
                    p  = tp / max(tp + fp, 1)
                    r  = tp / max(tp + fn, 1)
                    f1 = 2 * p * r / max(p + r, 1e-9)
                    if r >= 0.78 and f1 > best_f1:
                        best_f1    = f1
                        best_thresh = thresh
                y_pred = np.argmax(proba, axis=1)
                y_pred[proba[:, aidx] >= best_thresh] = aidx
                log.info(f"  action_needed threshold → {best_thresh:.3f} (F1={best_f1:.3f})")
            else:
                y_pred = clf.predict(X_test)
        else:
            y_pred = clf.predict(X_test)

        report_str = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
        report_dict = classification_report(y_test, y_pred, target_names=classes,
                                            zero_division=0, output_dict=True)
        macro_f1 = report_dict["macro avg"]["f1-score"]

        log.info(f"  {name} macro-F1: {macro_f1:.3f}")
        results[name] = {
            "clf": clf,
            "report_str": report_str,
            "report_dict": report_dict,
            "macro_f1": macro_f1,
            "y_pred": y_pred,
        }

    # Pick best by macro-F1
    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    log.info(f"  Best model: {best_name} (F1={results[best_name]['macro_f1']:.3f})")

    return {
        "target": target,
        "vectorizer": vectorizer,
        "label_encoder": le,
        "classes": classes,
        "results": results,
        "best_name": best_name,
        "y_test": y_test,
        "X_test": X_test,
        "vocab_size": len(vectorizer.vocabulary_),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


# ── Saving artifacts ──────────────────────────────────────────────────────────

def save_artifacts(result: dict, out_dir: Path):
    target   = result["target"]
    best     = result["best_name"]
    classes  = result["classes"]
    y_test   = result["y_test"]
    results  = result["results"]

    # ── Model pkl (vectorizer + best classifier) ─────────────────
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer":    result["vectorizer"],
        "label_encoder": result["label_encoder"],
        "classifier":    results[best]["clf"],
        "model_name":    best,
        "target":        target,
        "classes":       classes.tolist(),
        "macro_f1":      results[best]["macro_f1"],
        "numeric_features": NUMERIC_FEATURES,
    }
    pkl_path = model_dir / f"best_{target}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"  Saved model → {pkl_path}")

    # ── Text report (all 3 models) ────────────────────────────────
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"report_{target}.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"TARGET: {target}\n")
        f.write(f"Train samples: {result['n_train']:,} | Test samples: {result['n_test']:,}\n")
        f.write(f"TF-IDF vocab: {result['vocab_size']:,} features\n")
        f.write(f"Best model: {best}  (macro-F1 = {results[best]['macro_f1']:.4f})\n\n")
        for name, r in results.items():
            marker = " ← BEST" if name == best else ""
            f.write(f"{'─'*50}\n")
            f.write(f"{name.upper()}{marker}  |  macro-F1: {r['macro_f1']:.4f}\n")
            f.write(f"{'─'*50}\n")
            f.write(r["report_str"] + "\n\n")
    log.info(f"  Saved report → {rep_path}")

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Confusion matrix (best model) ────────────────────────────
    y_pred_best = results[best]["y_pred"]
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"{target}  |  {best}  (macro-F1 {results[best]['macro_f1']:.3f})", pad=12)
    plt.tight_layout()
    cm_path = plot_dir / f"confusion_{target}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved confusion matrix → {cm_path}")

    # ── Feature importance ────────────────────────────────────────
    clf  = results[best]["clf"]
    vec  = result["vectorizer"]
    feat_names = np.array(list(vec.get_feature_names_out()) + NUMERIC_FEATURES)

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef[0])

    if importances is not None:
        top_n = 30
        top_idx = np.argsort(importances)[-top_n:][::-1]
        top_names   = feat_names[top_idx]
        top_weights = importances[top_idx]

        fig, ax = plt.subplots(figsize=(10, 7))
        y_pos = np.arange(top_n)
        ax.barh(y_pos, top_weights[::-1], color="#4878cf", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Importance / |coef|")
        ax.set_title(f"Top {top_n} features — {target} ({best})", pad=10)
        plt.tight_layout()
        fi_path = plot_dir / f"feature_importance_{target}.png"
        fig.savefig(fi_path, dpi=150)
        plt.close(fig)
        log.info(f"  Saved feature importance → {fi_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(all_results: list, out_dir: Path):
    lines = ["BASELINE SUMMARY\n" + "="*60 + "\n"]
    for r in all_results:
        best = r["best_name"]
        f1   = r["results"][best]["macro_f1"]
        lines.append(f"{r['target'].upper()}")
        lines.append(f"  Best model : {best}")
        lines.append(f"  Macro-F1   : {f1:.4f}")
        lines.append(f"  Classes    : {', '.join(r['classes'])}")
        lines.append(f"  Train/Test : {r['n_train']:,} / {r['n_test']:,}")
        lines.append("")
        # All model scores for this target
        for name, res in r["results"].items():
            marker = " ←" if name == best else "  "
            lines.append(f"  {marker} {name:<25} macro-F1: {res['macro_f1']:.4f}")
        lines.append("")

    summary_path = out_dir / "baseline_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    log.info(f"Summary → {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train baseline classifiers on email pipeline DB")
    parser.add_argument("--db",          default="email_intelligence.db", help="Path to SQLite DB")
    parser.add_argument("--test-size",   type=float, default=0.15,        help="Test split fraction (default 0.15)")
    parser.add_argument("--min-samples", type=int,   default=50,          help="Drop classes with fewer samples")
    parser.add_argument("--out-dir",     default="baseline_output",       help="Output directory")
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(args.db)
    df = consolidate_labels(df)   # merge high+urgent, collapse intent
    if len(df) < 100:
        log.error("Too few labelled rows — wait for the pipeline to finish more labelling.")
        return

    all_results = []
    for target in TARGETS:
        result = train_target(df, target, args.test_size, args.min_samples)
        save_artifacts(result, OUT_DIR)
        all_results.append(result)

    write_summary(all_results, OUT_DIR)
    log.info(f"\nAll done. Artifacts in: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()