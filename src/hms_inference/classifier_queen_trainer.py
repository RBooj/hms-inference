from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDINGS_ROOT = Path.cwd() / "data" / "embeddings"
MODELS_ROOT = Path.cwd() / "data" / "models"
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

INPUT_DIM = 768
HIDDEN_DIM = 128
DROPOUT = 0.2

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 20
PATIENCE = 4

class QueenClassifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
                nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def list_shard_prefixes(split_dir: Path) -> list[Path]:
    meta_files = sorted(split_dir.glob("part_*_meta.parquet"))
    prefixes = []
    for meta_path in meta_files:
        prefix_str = meta_path.name.replace("_meta.parquet", "")
        prefixes.append(split_dir / prefix_str)
    return prefixes

def load_shard(prefix: Path) -> tuple[np.ndarray, np.ndarray]:
    x_path = prefix.parent / f"{prefix.name}_embeddings.npy"
    y_path = prefix.parent / f"{prefix.name}_labels.npy"

    X = np.load(x_path)
    y = np.load(y_path)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D embedding array in {x_path}, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected 1D label array in {y_path}, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"Shard size mismatch: {x_path} has {len(X)} rows, {y_path} has {len(y)} rows")

    return X.astype(np.float32), y.astype(np.float32)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_pos_weight(train_dir: Path) -> torch.Tensor:
    total_pos = 0
    total_neg = 0

    for prefix in list_shard_prefixes(train_dir):
        _, y = load_shard(prefix)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        total_pos += pos
        total_neg += neg

    if total_pos == 0:
        raise ValueError("No positive samples found in training set.")
    if total_neg == 0:
        raise ValueError("No negative samples found in training set.")

    # BCEWithLogitsLoss pos_weight > 1 increases weight on positives.
    # Here positives are queen_present=True (label 1).
    pos_weight = total_neg / total_pos
    return torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_count = 0

    all_probs = []
    all_preds = []
    all_targets = []

    for xb, yb in val_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = xb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_targets.append(yb.detach().cpu())

    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    accuracy = float((preds == targets).mean())

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    avg_loss = total_loss / total_count if total_count > 0 else math.nan

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def load_full_split(split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    all_x = []
    all_y = []

    for prefix in list_shard_prefixes(split_dir):
        X, y = load_shard(prefix)
        all_x.append(X)
        all_y.append(y)

    if not all_x:
        raise ValueError(f"No shards found in {split_dir}")

    X = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    return X, y


def train_one_epoch(
    model: nn.Module,
    train_dir: Path,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    shard_prefixes = list_shard_prefixes(train_dir)
    if not shard_prefixes:
        raise ValueError(f"No training shards found in {train_dir}")

    for shard_idx, prefix in enumerate(shard_prefixes, start=1):
        X, y = load_shard(prefix)
        train_loader = make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True)

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        print(f"[Train] Finished shard {shard_idx}/{len(shard_prefixes)}: {prefix.name}")

    return total_loss / total_count if total_count > 0 else math.nan


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

    print(f"[Train] Device: {DEVICE}")

    print("[Train] Loading validation embeddings...")
    X_val, y_val = load_full_split(val_dir)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[Train] Validation rows: {len(y_val)}")

    print("[Train] Loading test embeddings...")
    X_test, y_test = load_full_split(test_dir)
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[Train] Test rows: {len(y_test)}")

    print("[Train] Computing class weight from training shards...")
    pos_weight = compute_pos_weight(train_dir)
    print(f"[Train] pos_weight: {float(pos_weight.item()):.4f}")

    model = QueenClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    best_model_path = MODELS_ROOT / "queen_classifier_best.pt"
    metrics_path = MODELS_ROOT / "queen_classifier_metrics.json"

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n[Train] Epoch {epoch}/{MAX_EPOCHS}")

        train_loss = train_one_epoch(
            model=model,
            train_dir=train_dir,
            optimizer=optimizer,
            criterion=criterion,
        )

        val_metrics = evaluate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
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
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": best_val_f1,
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

    print(f"\n[Train] Best epoch: {best_epoch}, best val_f1: {best_val_f1:.4f}")

    print("[Train] Loading best model for final test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        val_loader=test_loader,
        criterion=criterion,
    )

    print(
        f"[Test] loss={test_metrics['loss']:.4f} | "
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
        "best_val_f1": best_val_f1,
        "history": history,
        "test_metrics": test_metrics,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[Train] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

