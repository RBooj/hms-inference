from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_ROOT = Path.cwd() / "data" / "embeddings"
MODELS_ROOT = Path.cwd() / "data" / "models"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

INPUT_DIM = 768
HIDDEN_DIM = 512
DROPOUT = 0.1

BATCH_SIZE = 512
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 25
PATIENCE = 5

DEFAULT_THRESHOLD = 0.5

RECENCY_TAU_DAYS = 7.0
MIN_RECENCY_WEIGHT = 0.25
USE_RECENCY_WEIGHTING = False


class QueenClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def compute_recency_weights(
    days_since: torch.Tensor,
    tau_days: float = RECENCY_TAU_DAYS,
    min_weight: float = MIN_RECENCY_WEIGHT,
) -> torch.Tensor:
    weights = torch.exp(-days_since / tau_days)
    weights = torch.clamp(weights, min=min_weight, max=1.0)
    return weights

def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    days_since: torch.Tensor,
    negative_weight: float,
    positive_weight: float,
    use_recency_weighting: bool = USE_RECENCY_WEIGHTING,
) -> torch.Tensor:
    base_loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    class_weights = torch.where(
        targets == 0,
        torch.full_like(targets, negative_weight),
        torch.full_like(targets, positive_weight),
    )

    if use_recency_weighting:
        recency_weights = compute_recency_weights(days_since)
    else:
        recency_weights = torch.ones_like(targets)

    total_weights = class_weights * recency_weights
    return (base_loss * total_weights).mean()

def list_shard_prefixes(split_dir: Path) -> list[Path]:
    meta_files = sorted(split_dir.glob("part_*_meta.parquet"))
    prefixes = []
    for meta_path in meta_files:
        prefix_str = meta_path.name.replace("_meta.parquet", "")
        prefixes.append(split_dir / prefix_str)
    return prefixes

def load_shard(prefix: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_path = prefix.parent / f"{prefix.name}_embeddings.npy"
    y_path = prefix.parent / f"{prefix.name}_labels.npy"
    meta_path = prefix.parent / f"{prefix.name}_meta.parquet"

    X = np.load(x_path)
    y = np.load(y_path)
    meta_df = pd.read_parquet(meta_path)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding array in {x_path}, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D label array in {y_path}, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"Shard size mismatch: {x_path} has {len(X)} rows, {y_path} has {len(y)} rows")
    if len(meta_df) != len(y):
        raise ValueError(
            f"Shard size mismatch: {meta_path} has {len(meta_df)} rows, {y_path} has {len(y)} rows"
        )

    if "days_since_inspection" not in meta_df.columns:
        raise ValueError(f"{meta_path} is missing 'days_since_inspection'")

    days_since = meta_df["days_since_inspection"].to_numpy(dtype=np.float32)

    return X.astype(np.float32), y.astype(np.float32), days_since

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

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    days_since: np.ndarray,
    batch_size: int,
    shuffle: bool,
    balanced: bool = False,
) -> DataLoader:
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    d_tensor = torch.from_numpy(days_since)
    ds = TensorDataset(x_tensor, y_tensor, d_tensor)

    if balanced:
        class_counts = np.bincount(y.astype(np.int64), minlength=2)

        if class_counts[0] == 0 or class_counts[1] == 0:
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        sample_weights = np.where(y == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    negative_weight: float,
    positive_weight: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_probs = []
    all_targets = []

    for xb, yb, db in val_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        db = db.to(DEVICE, non_blocking=True)

        logits = model(xb)
        loss = weighted_bce_loss(
            logits,
            yb,
            db,
            negative_weight=negative_weight,
            positive_weight=positive_weight,
        )

        probs = torch.sigmoid(logits)

        batch_size = xb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        all_probs.append(probs.detach().cpu())
        all_targets.append(yb.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= threshold).astype(np.float32)

    avg_loss = total_loss / total_count if total_count > 0 else math.nan
    metrics = compute_classification_metrics(preds, targets)
    metrics["loss"] = avg_loss

    metrics["prob_mean"] = float(probs.mean())
    metrics["prob_std"] = float(probs.std())
    metrics["pred_pos_rate"] = float(preds.mean())

    return metrics

def load_full_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_x = []
    all_y = []
    all_days = []

    for prefix in list_shard_prefixes(split_dir):
        X, y, days_since = load_shard(prefix)
        all_x.append(X)
        all_y.append(y)
        all_days.append(days_since)

    if not all_x:
        raise ValueError(f"No shards found in {split_dir}")

    X = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    days_since = np.concatenate(all_days, axis=0).astype(np.float32)
    return X, y, days_since

def train_one_epoch(
    model: nn.Module,
    train_dir: Path,
    optimizer: torch.optim.Optimizer,
    negative_weight: float,
    positive_weight: float,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    shard_prefixes = list_shard_prefixes(train_dir)
    if not shard_prefixes:
        raise ValueError(f"No training shards found in {train_dir}")

    for shard_idx, prefix in enumerate(shard_prefixes, start=1):
        X, y, days_since = load_shard(prefix)
        train_loader = make_loader(
            X,
            y,
            days_since,
            batch_size=BATCH_SIZE,
            shuffle=False,
            balanced=True,
        )

        for xb, yb, db in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            db = db.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = weighted_bce_loss(
                logits,
                yb,
                db,
                negative_weight=negative_weight,
                positive_weight=positive_weight,
            )

            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        print(f"[Train] Finished shard {shard_idx}/{len(shard_prefixes)}: {prefix.name}")

    return total_loss / total_count if total_count > 0 else math.nan

@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs = []
    all_targets = []

    for xb, yb, _ in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(yb.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    return probs, targets


def metrics_at_threshold(
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
    results = []
    for threshold in thresholds:
        results.append(metrics_at_threshold(probs, targets, threshold))
    return results


def choose_best_threshold_by_balanced_accuracy(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No threshold results to choose from.")
    return max(results, key=lambda r: r["balanced_accuracy"])

def compute_class_weights(train_dir: Path) -> dict[str, float]:
    total_pos = 0
    total_neg = 0

    for prefix in list_shard_prefixes(train_dir):
        _, y, _ = load_shard(prefix)
        total_pos += int((y == 1).sum())
        total_neg += int((y == 0).sum())

    if total_pos == 0:
        raise ValueError("No positive samples found in training set.")
    if total_neg == 0:
        raise ValueError("No negative samples found in training set.")

    total = total_pos + total_neg
    positive_weight = total / (2.0 * total_pos)
    negative_weight = total / (2.0 * total_neg)

    return {
        "positive_count": total_pos,
        "negative_count": total_neg,
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
    }


def main() -> None:
    train_dir = EMBEDDINGS_ROOT / "queen_train"
    val_dir = EMBEDDINGS_ROOT / "queen_val"
    test_dir = EMBEDDINGS_ROOT / "queen_test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training embedding dir not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation embedding dir not found: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test embedding dir not found: {test_dir}")

    class_stats = compute_class_weights(train_dir)
    negative_weight = class_stats["negative_weight"]
    positive_weight = class_stats["positive_weight"]

    print(
            "[Train] Class balance: "
            f"neg={class_stats['negative_count']} "
            f"pos={class_stats['positive_count']} | "
            f"negative_weight={negative_weight:.4f} "
            f"positive_weight={positive_weight:.4f}"
            )

    print(
            "[Train] Recency weighting: "
            f"enabled={USE_RECENCY_WEIGHTING} | "
            f"tau_days={RECENCY_TAU_DAYS:.2f} | "
            f"min_weight={MIN_RECENCY_WEIGHT:.2f}"
            )

    print(f"[Train] Device: {DEVICE}")

    print("[Train] Loading validation embeddings...")
    X_val, y_val, d_val = load_full_split(val_dir)
    val_loader = make_loader(X_val, y_val, d_val, batch_size=BATCH_SIZE, shuffle=False, balanced=False)
    print(f"[Train] Validation rows: {len(y_val)}")

    print("[Train] Loading test embeddings...")
    X_test, y_test, d_test = load_full_split(test_dir)
    test_loader = make_loader(X_test, y_test, d_test, batch_size=BATCH_SIZE, shuffle=False, balanced=False)
    print(f"[Train] Test rows: {len(y_test)}")

    model = QueenClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_bal_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    best_model_path = MODELS_ROOT / "queen_classifier_best.pt"
    metrics_path = MODELS_ROOT / "queen_classifier_metrics.json"

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n[Train] Epoch {epoch}/{MAX_EPOCHS}")

        first_weight_before = model.net[0].weight.detach().clone()

        train_loss = train_one_epoch(
            model=model,
            train_dir=train_dir,
            optimizer=optimizer,
            negative_weight=negative_weight,
            positive_weight=positive_weight,
        )

        first_weight_after = model.net[0].weight.detach()
        weight_delta = float((first_weight_after - first_weight_before).abs().mean().item())
        print(f"[Train] mean abs delta first layer weights = {weight_delta:.8f}")

        val_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            negative_weight=negative_weight,
            positive_weight=positive_weight,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(epoch_record)

        print(
            f"[Train] train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_specificity={val_metrics['specificity']:.4f} | "
            f"prob_mean={val_metrics['prob_mean']:.4f} | "
            f"prob_std={val_metrics['prob_std']:.4f} | "
            f"pred_pos_rate={val_metrics['pred_pos_rate']:.4f}"
        )

        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_balanced_accuracy": best_val_bal_acc,
                    "config": {
                        "input_dim": INPUT_DIM,
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                    },
                },
                best_model_path,
            )

            print(f"[Train] New best model saved to {best_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"[Train] Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    print(f"\n[Train] Best epoch: {best_epoch}, best val_bal_acc: {best_val_bal_acc:.4f}")

    print("[Train] Loading best model for final test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("[Train] Collecting validation probabilities for threshold tuning...")
    val_probs, val_targets = predict_probabilities(model, val_loader)

    thresholds = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
    val_threshold_results = scan_thresholds(val_probs, val_targets, thresholds)

    print("\n[Validation threshold scan]")
    for r in val_threshold_results:
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

    best_threshold_result = choose_best_threshold_by_balanced_accuracy(val_threshold_results)
    best_threshold = best_threshold_result["threshold"]

    print(
        f"\n[Train] Best validation threshold by balanced accuracy: {best_threshold:.2f} | "
        f"bal_acc={best_threshold_result['balanced_accuracy']:.4f} | "
        f"f1={best_threshold_result['f1']:.4f} | "
        f"precision={best_threshold_result['precision']:.4f} | "
        f"recall={best_threshold_result['recall']:.4f} | "
        f"specificity={best_threshold_result['specificity']:.4f}"
    )


    print("[Train] Collecting test probabilities...")
    test_probs, test_targets = predict_probabilities(model, test_loader)

    test_metrics = metrics_at_threshold(
        test_probs,
        test_targets,
        threshold=best_threshold,
    )

    print(
        f"[Test @ threshold={best_threshold:.2f}] "
        f"acc={test_metrics['accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | "
        f"precision={test_metrics['precision']:.4f} | "
        f"recall={test_metrics['recall']:.4f}"
    )
    print(
        f"[Test] confusion matrix: "
        f"TP={test_metrics['tp']} TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} FN={test_metrics['fn']}"
    )

    metrics_payload = {
        "best_epoch": best_epoch,
        "val_balanced_accuracy": best_val_bal_acc,
        "selected_threshold": best_threshold,
        "validation_threshold_results": val_threshold_results,
        "history": history,
        "test_metrics": test_metrics,
    }


    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[Train] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

