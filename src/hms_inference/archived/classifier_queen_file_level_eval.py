from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path.cwd()
EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"
MODELS_ROOT = PROJECT_ROOT / "data" / "models"

BATCH_SIZE = 1024


class QueenClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def list_shard_prefixes(split_dir: Path) -> list[Path]:
    meta_files = sorted(split_dir.glob("part_*_meta.parquet"))
    prefixes = []
    for meta_path in meta_files:
        prefix_str = meta_path.name.replace("_meta.parquet", "")
        prefixes.append(split_dir / prefix_str)
    return prefixes


def load_shard(prefix: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_path = prefix.parent / f"{prefix.name}_meta.parquet"
    x_path = prefix.parent / f"{prefix.name}_embeddings.npy"
    y_path = prefix.parent / f"{prefix.name}_labels.npy"

    meta = pd.read_parquet(meta_path)
    X = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.float32)

    if len(meta) != len(X) or len(meta) != len(y):
        raise ValueError(
            f"Shard length mismatch for {prefix.name}: "
            f"meta={len(meta)}, X={len(X)}, y={len(y)}"
        )

    return meta, X, y


def load_full_split(split_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metas = []
    xs = []
    ys = []

    prefixes = list_shard_prefixes(split_dir)
    if not prefixes:
        raise ValueError(f"No shards found in {split_dir}")

    for prefix in prefixes:
        meta, X, y = load_shard(prefix)
        metas.append(meta)
        xs.append(X)
        ys.append(y)

    meta_df = pd.concat(metas, ignore_index=True)
    X_all = np.concatenate(xs, axis=0).astype(np.float32)
    y_all = np.concatenate(ys, axis=0).astype(np.float32)

    return meta_df, X_all, y_all


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


@torch.no_grad()
def predict_probabilities(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(yb.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    return probs, targets


def compute_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> dict:
    accuracy = float((preds == targets).mean())

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def metrics_from_probs(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict:
    preds = (probs >= threshold).astype(np.float32)
    metrics = compute_classification_metrics(preds, targets)
    metrics["threshold"] = threshold
    return metrics


def scan_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    thresholds: list[float],
) -> list[dict]:
    return [metrics_from_probs(probs, targets, thr) for thr in thresholds]


def choose_best_threshold_by_balanced_accuracy(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No threshold results to choose from.")
    return max(results, key=lambda r: r["balanced_accuracy"])


def top_k_mean(values: np.ndarray, k: int) -> float:
    if len(values) == 0:
        return float("nan")
    k = min(k, len(values))
    # sort descending and average top-k
    topk = np.sort(values)[-k:]
    return float(np.mean(topk))


def aggregate_file_predictions(
    meta_df: pd.DataFrame,
    probs: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    df = meta_df.copy()
    df["chunk_prob"] = probs
    df["chunk_label"] = labels.astype(np.int64)

    # sanity check: all chunks from same wav should share one label
    label_consistency = df.groupby("wav_path")["chunk_label"].nunique()
    bad = label_consistency[label_consistency > 1]
    if not bad.empty:
        raise ValueError(
            f"Found wav_path values with inconsistent labels across chunks: {bad.index.tolist()[:5]}"
        )

    rows = []

    for wav_path, group in df.groupby("wav_path", sort=False):
        chunk_probs = group["chunk_prob"].to_numpy(dtype=np.float32)
        queen_present = int(group["chunk_label"].iloc[0])

        rows.append(
            {
                "wav_path": wav_path,
                "queen_present": queen_present,
                "n_chunks": len(group),
                "mean_prob": float(np.mean(chunk_probs)),
                "median_prob": float(np.median(chunk_probs)),
                "max_prob": float(np.max(chunk_probs)),
                "top5_mean_prob": top_k_mean(chunk_probs, 5),
                "top10_mean_prob": top_k_mean(chunk_probs, 10),
            }
        )

    return pd.DataFrame(rows)


def evaluate_score_column(
    df: pd.DataFrame,
    score_col: str,
    thresholds: list[float],
    label_col: str = "queen_present",
) -> tuple[list[dict], dict]:
    results = scan_thresholds(
        df[score_col].to_numpy(dtype=np.float32),
        df[label_col].to_numpy(dtype=np.float32),
        thresholds,
    )
    best = choose_best_threshold_by_balanced_accuracy(results)
    return results, best


def main() -> None:
    val_dir = EMBEDDINGS_ROOT / "queen_val"
    test_dir = EMBEDDINGS_ROOT / "queen_test"
    model_path = MODELS_ROOT / "queen_classifier_best.pt"
    output_path = MODELS_ROOT / "queen_file_level_metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}")

    print(f"[File Eval] Device: {DEVICE}")
    print("[File Eval] Loading best trained model checkpoint...")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = QueenClassifier(
        input_dim=checkpoint["config"]["input_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[File Eval] Loading validation embeddings + metadata...")
    val_meta, X_val, y_val = load_full_split(val_dir)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE)
    val_probs, val_targets = predict_probabilities(model, val_loader)

    print(f"[File Eval] Validation chunk rows: {len(val_meta)}")
    print(f"[File Eval] Validation unique wavs: {val_meta['wav_path'].nunique()}")

    val_file_df = aggregate_file_predictions(val_meta, val_probs, val_targets)
    print(f"[File Eval] Validation file rows: {len(val_file_df)}")
    print("[File Eval] Validation file label counts:")
    print(val_file_df["queen_present"].value_counts(dropna=False))

    thresholds = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]

    score_columns = [
        "mean_prob",
        "median_prob",
        "max_prob",
        "top5_mean_prob",
        "top10_mean_prob",
    ]

    validation_results_by_method = {}
    best_thresholds_by_method = {}

    for score_col in score_columns:
        print(f"\n[Validation file-level threshold scan | {score_col}]")
        results, best = evaluate_score_column(
            val_file_df,
            score_col=score_col,
            thresholds=thresholds,
        )

        validation_results_by_method[score_col] = results
        best_thresholds_by_method[score_col] = best

        for r in results:
            print(
                f"thr={r['threshold']:.2f} | "
                f"acc={r['accuracy']:.4f} | "
                f"bal_acc={r['balanced_accuracy']:.4f} | "
                f"f1={r['f1']:.4f} | "
                f"precision={r['precision']:.4f} | "
                f"recall={r['recall']:.4f} | "
                f"specificity={r['specificity']:.4f} | "
                f"TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}"
            )

        print(
            f"\n[File Eval] Best {score_col} threshold by balanced accuracy: {best['threshold']:.2f} | "
            f"bal_acc={best['balanced_accuracy']:.4f} | "
            f"f1={best['f1']:.4f} | "
            f"precision={best['precision']:.4f} | "
            f"recall={best['recall']:.4f} | "
            f"specificity={best['specificity']:.4f}"
        )

    print("\n[File Eval] Loading test embeddings + metadata...")
    test_meta, X_test, y_test = load_full_split(test_dir)
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE)
    test_probs, test_targets = predict_probabilities(model, test_loader)

    print(f"[File Eval] Test chunk rows: {len(test_meta)}")
    print(f"[File Eval] Test unique wavs: {test_meta['wav_path'].nunique()}")

    test_file_df = aggregate_file_predictions(test_meta, test_probs, test_targets)
    print(f"[File Eval] Test file rows: {len(test_file_df)}")
    print("[File Eval] Test file label counts:")
    print(test_file_df["queen_present"].value_counts(dropna=False))

    test_metrics_by_method = {}

    for score_col in score_columns:
        best = best_thresholds_by_method[score_col]

        metrics = metrics_from_probs(
            test_file_df[score_col].to_numpy(dtype=np.float32),
            test_file_df["queen_present"].to_numpy(dtype=np.float32),
            threshold=best["threshold"],
        )

        test_metrics_by_method[score_col] = metrics

        print(
            f"\n[Test | {score_col} @ thr={best['threshold']:.2f}] "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"specificity={metrics['specificity']:.4f}"
        )
        print(
            f"[Test | {score_col}] confusion matrix: "
            f"TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']}"
        )

    payload = {
        "best_checkpoint_epoch": checkpoint.get("epoch"),
        "best_checkpoint_val_balanced_accuracy": checkpoint.get("val_balanced_accuracy"),
        "validation_results_by_method": validation_results_by_method,
        "best_thresholds_by_method": best_thresholds_by_method,
        "test_metrics_by_method": test_metrics_by_method,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[File Eval] Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
