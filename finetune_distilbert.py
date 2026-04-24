"""
finetune_distilbert.py
──────────────────────
Fine-tunes DistilBERT on 3 targets: priority, sender_type, intent
Hardware: CUDA (RTX 3050 6GB) — batch_size and max_length tuned for 6GB VRAM

Outputs (all in ./distilbert_output/):
  models/   — distilbert_<target>/   (HuggingFace saved model + tokenizer)
  models/   — bundle_<target>.pkl    (drop-in replacement for train_baseline pkls)
  reports/  — report_<target>.txt
  plots/    — confusion_<target>.png
  distilbert_summary.txt

Usage:
  python finetune_distilbert.py
  python finetune_distilbert.py --db D:\\mailmind\\email_intelligence.db
  python finetune_distilbert.py --epochs 4 --batch-size 16
  python finetune_distilbert.py --target priority   # single target only
"""

import argparse
import logging
import pickle
import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "distilbert-base-uncased"
TARGETS      = ["priority", "sender_type", "intent"]
MAX_LENGTH   = 256      # truncate at 256 tokens — safe for 6GB VRAM
BATCH_SIZE   = 16       # train batch — fits RTX 3050 6GB
EVAL_BATCH   = 32       # eval doesn't need gradients, can go bigger
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.1
TEST_SIZE    = 0.15
MIN_SAMPLES  = 50
OUT_DIR      = Path("distilbert_output")

# Label consolidation maps (same as train_baseline)
PRIORITY_MAP = {
    "urgent": "action_needed",
    "high":   "action_needed",
    "normal": "normal",
    "low":    "low",
}

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

SENDER_EXCLUDE = {"unknown", "external", "government", "system"}

# Class weights for imbalanced targets
CLASS_BOOST = {
    "action_needed": 4.0, "high": 2.0,   "normal": 1.0, "low": 0.8,
    "client":        5.0, "personal": 2.0, "spam": 1.5,
    "vendor":        1.0, "internal": 0.8,
    "action":        2.0, "info": 1.0,    "other": 1.5,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
        SELECT
            ce.subject,
            ce.body,
            ce.body_length,
            CAST(ce.is_reply AS INTEGER) AS is_reply,
            ce.cc_count,
            rc.priority,
            rc.sender_type,
            rc.intent
        FROM cleaned_emails ce
        JOIN results_classifications rc ON rc.email_id = ce.id
        WHERE ce.body IS NOT NULL
          AND ce.body != ''
          AND rc.priority    IS NOT NULL
          AND rc.sender_type NOT IN ('unknown','external','government','system')
          AND rc.intent      IS NOT NULL
    """, conn)
    conn.close()

    # Combine subject + body into one input text
    df["text"] = (
        df["subject"].fillna("").str.strip() + " [SEP] " +
        df["body"].fillna("").str.strip().str[:1500]  # first 1500 chars of body
    )

    # Consolidate labels
    df["priority"]    = df["priority"].map(PRIORITY_MAP).fillna("normal")
    df["intent"]      = df["intent"].map(INTENT_MAP).fillna("other")

    log.info(f"Loaded {len(df):,} rows from {db_path}")
    return df


def drop_rare(df: pd.DataFrame, target: str, min_samples: int) -> pd.DataFrame:
    counts = df[target].value_counts()
    keep = counts[counts >= min_samples].index
    dropped = counts[counts < min_samples]
    if len(dropped):
        log.warning(f"[{target}] Dropping rare: " +
                    ", ".join(f"{c}({n})" for c, n in dropped.items()))
    return df[df[target].isin(keep)].copy()


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_target(df: pd.DataFrame, target: str, tokenizer, device: torch.device,
                     epochs: int, batch_size: int) -> dict:
    log.info(f"{'='*60}")
    log.info(f"TARGET: {target.upper()}")
    log.info(f"{'='*60}")

    df_t = drop_rare(df, target, MIN_SAMPLES)
    log.info(f"Distribution:\n{df_t[target].value_counts().to_string()}")

    le = LabelEncoder()
    y  = le.fit_transform(df_t[target])
    classes = le.classes_
    n_classes = len(classes)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(sss.split(df_t, y))

    texts_train = df_t.iloc[train_idx]["text"].tolist()
    texts_test  = df_t.iloc[test_idx]["text"].tolist()
    y_train     = y[train_idx]
    y_test      = y[test_idx]

    log.info(f"Train: {len(y_train):,} | Test: {len(y_test):,}")

    # Build datasets
    log.info("Tokenizing...")
    train_ds = EmailDataset(texts_train, y_train, tokenizer, MAX_LENGTH)
    test_ds  = EmailDataset(texts_test,  y_test,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=EVAL_BATCH, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Class weights tensor for loss
    class_weights = torch.tensor(
        [CLASS_BOOST.get(c, 1.0) for c in classes],
        dtype=torch.float,
        device=device,
    )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    best_val_f1  = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                log.info(f"  Epoch {epoch} step {step+1}/{len(train_loader)} "
                         f"loss={total_loss/(step+1):.4f}")

        # Validation
        val_f1, val_report_str, val_report_dict, y_pred = evaluate(
            model, test_loader, le, classes, device
        )
        log.info(f"  Epoch {epoch} val macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_report_str  = val_report_str
            best_report_dict = val_report_dict
            best_y_pred      = y_pred

    # Restore best checkpoint
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    log.info(f"  Best macro-F1: {best_val_f1:.4f}")

    return {
        "target":        target,
        "model":         model,
        "tokenizer":     tokenizer,
        "label_encoder": le,
        "classes":       classes,
        "macro_f1":      best_val_f1,
        "report_str":    best_report_str,
        "report_dict":   best_report_dict,
        "y_pred":        best_y_pred,
        "y_test":        y_test,
        "n_train":       len(y_train),
        "n_test":        len(y_test),
    }


def evaluate(model, loader, le, classes, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_pred = np.array(all_preds)
    y_test = np.array(all_labels)

    report_str  = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=classes,
                                        zero_division=0, output_dict=True)
    macro_f1 = report_dict["macro avg"]["f1-score"]
    return macro_f1, report_str, report_dict, y_pred


# ── Save artifacts ────────────────────────────────────────────────────────────

def save_artifacts(result: dict, out_dir: Path):
    target  = result["target"]
    classes = result["classes"]
    y_test  = result["y_test"]
    y_pred  = result["y_pred"]
    model   = result["model"]

    # Save HuggingFace model + tokenizer
    model_dir = out_dir / "models" / f"distilbert_{target}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    result["tokenizer"].save_pretrained(model_dir)
    log.info(f"  HF model → {model_dir}")

    # Save pkl bundle (same structure as train_baseline for Dev B compatibility)
    bundle = {
        "model_dir":     str(model_dir),
        "label_encoder": result["label_encoder"],
        "target":        target,
        "classes":       classes.tolist(),
        "macro_f1":      result["macro_f1"],
        "model_type":    "distilbert",
    }
    pkl_path = out_dir / "models" / f"bundle_{target}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"  Bundle pkl → {pkl_path}")

    # Report
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_path = rep_dir / f"report_{target}.txt"
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"TARGET: {target}\n")
        f.write(f"Model: {MODEL_NAME} fine-tuned\n")
        f.write(f"Train: {result['n_train']:,} | Test: {result['n_test']:,}\n")
        f.write(f"Best epoch macro-F1: {result['macro_f1']:.4f}\n\n")
        f.write(result["report_str"])
    log.info(f"  Report → {rep_path}")

    # Confusion matrix
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=30)
    ax.set_title(f"{target} | distilbert (macro-F1 {result['macro_f1']:.3f})", pad=12)
    plt.tight_layout()
    cm_path = plot_dir / f"confusion_{target}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log.info(f"  Confusion matrix → {cm_path}")


def write_summary(all_results: list, out_dir: Path):
    lines = ["DISTILBERT SUMMARY\n" + "="*60 + "\n"]
    for r in all_results:
        rd = r["report_dict"]
        lines.append(f"{r['target'].upper()}")
        lines.append(f"  Model      : {MODEL_NAME} fine-tuned")
        lines.append(f"  Macro-F1   : {r['macro_f1']:.4f}")
        lines.append(f"  Weighted-F1: {rd['weighted avg']['f1-score']:.4f}")
        lines.append(f"  Accuracy   : {rd['accuracy']:.4f}")
        lines.append(f"  Classes    : {', '.join(r['classes'])}")
        lines.append(f"  Train/Test : {r['n_train']:,} / {r['n_test']:,}")
        lines.append("")
        # Per-class breakdown
        for cls in r["classes"]:
            if cls in rd:
                lines.append(
                    f"  {cls:<20} "
                    f"P={rd[cls]['precision']:.2f}  "
                    f"R={rd[cls]['recall']:.2f}  "
                    f"F1={rd[cls]['f1-score']:.2f}  "
                    f"n={int(rd[cls]['support'])}"
                )
        lines.append("")

    summary_path = out_dir / "distilbert_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    log.info(f"Summary -> {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on email pipeline DB")
    parser.add_argument("--db",         default="email_intelligence.db")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--out-dir",    default="distilbert_output")
    parser.add_argument("--target",     choices=TARGETS, default=None,
                        help="Train a single target only (omit for all three)")
    args = parser.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                 f"VRAM free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.1f}GB")

    # Load tokenizer once — shared across all targets
    log.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data once
    df = load_data(args.db)
    if len(df) < 100:
        log.error("Too few rows.")
        return

    targets = [args.target] if args.target else TARGETS
    all_results = []

    for target in targets:
        result = train_one_target(
            df, target, tokenizer, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        save_artifacts(result, OUT_DIR)
        all_results.append(result)

        # Free VRAM between targets
        del result["model"]
        torch.cuda.empty_cache()

    write_summary(all_results, OUT_DIR)
    log.info(f"\nAll done. Artifacts in: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()