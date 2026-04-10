from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from hms_inference.queen_audio_dataset import (
    QueenAudioDataset,
    collate_queen_audio,
    compute_ast_stats,
)
from hms_inference.queen_ast_model import ASTQueenClassifier
from hms_inference.config_loader import QueenPipelineConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SPLITS_DIR = Path.cwd() / "data" / "splits"
# MODELS_ROOT = Path.cwd() / "data" / "models"
# MODELS_ROOT.mkdir(parents=True, exist_ok=True)
#
# MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.448"
#
# BATCH_SIZE = 5
# MAX_EPOCHS = 10
# PATIENCE = 3
#
# HEAD_LEARNING_RATE = 5e-5
# BACKBONE_LEARNING_RATE = 5e-6
# WEIGHT_DECAY = 1e-4
#
# DROPOUT = 0.2
# DEFAULT_THRESHOLD = 0.7
#
# FREEZE_STRATEGY = "frozen"
# UNFREEZE_LAST_N = 1
#
# USE_CLASS_WEIGHTING = True
# USE_BALANCED_SAMPLER = True
#
# GRAD_CLIP_NORM = 1.0
# USE_AMP = torch.cuda.is_available()
#
# NUM_WORKERS = 6
# PIN_MEMORY = torch.cuda.is_available()
# LOG_EVERY_N_BATCHES = 10
#
# FORCE_RECOMPUTE_NORMALIZATIONS = False
# DO_NORMALIZE = False


def format_gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "gpu_mem=n/a"
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    return f"gpu_alloc={allocated:.2f}GB reserved={reserved:.2f}GB"


def compute_classification_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    accuracy = float((preds == targets).mean())

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

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


def metrics_at_threshold(
    probs: np.ndarray, targets: np.ndarray, threshold: float
) -> dict:
    preds = (probs >= threshold).astype(np.float32)
    metrics = compute_classification_metrics(preds, targets)
    metrics["threshold"] = threshold
    return metrics


def scan_thresholds(
    probs: np.ndarray, targets: np.ndarray, thresholds: list[float]
) -> list[dict]:
    return [metrics_at_threshold(probs, targets, t) for t in thresholds]


def choose_best_threshold_by_balanced_accuracy(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No threshold results found")
    return max(results, key=lambda r: r["balanced_accuracy"])


def make_balanced_sampler(dataset: QueenAudioDataset) -> WeightedRandomSampler:
    labels = dataset.df["queen_present"].astype(bool).astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=2)

    if class_counts[0] == 0 or class_counts[1] == 0:
        raise ValueError("Balanced sampler requires both classes in dataset")

    sample_weights = np.where(labels == 1, 1.0 / class_counts[1], 1.0 / class_counts[0])
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.float32)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_pos_weight(dataset: QueenAudioDataset) -> torch.Tensor:
    labels = dataset.df["queen_present"].astype(bool).astype(int).to_numpy()
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())

    if pos == 0 or neg == 0:
        raise ValueError("Training set must contain both classes")

    # BCEWithLogitsLoss(pos_weight=neg/pos)
    return torch.tensor(neg / pos, dtype=torch.float32, device=DEVICE)


def build_loaders(
    cfg: QueenPipelineConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, QueenAudioDataset]:
    train_ds = QueenAudioDataset(
        cfg.paths.splits_dir / cfg.project.name / "queen_train.parquet",
        cache_waveforms=cfg.audio.cache_waveforms,
        target_sample_rate=cfg.audio.target_sample_rate,
    )
    val_ds = QueenAudioDataset(
        cfg.paths.splits_dir / cfg.project.name / "queen_val.parquet",
        cache_waveforms=cfg.audio.cache_waveforms,
        target_sample_rate=cfg.audio.target_sample_rate,
    )
    test_ds = QueenAudioDataset(
        cfg.paths.splits_dir / cfg.project.name / "queen_test.parquet",
        cache_waveforms=cfg.audio.cache_waveforms,
        target_sample_rate=cfg.audio.target_sample_rate,
    )

    print(f"[Data] Train rows: {len(train_ds)}")
    print(f"[Data] Val rows:   {len(val_ds)}")
    print(f"[Data] Test rows:  {len(test_ds)}")

    print("[Data] Train label counts:")
    print(train_ds.df["queen_present"].value_counts(dropna=False))

    train_loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        collate_fn=collate_queen_audio,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
    )

    eval_loader_kwargs = dict(
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_queen_audio,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
    )

    if cfg.imbalance.use_balanced_sampler:
        train_loader = DataLoader(
            train_ds,
            sampler=make_balanced_sampler(train_ds),
            **train_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **train_loader_kwargs,
        )

    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)

    print(f"[Data] Train batches/epoch: {len(train_loader)}")
    print(f"[Data] Val batches:         {len(val_loader)}")
    print(f"[Data] Test batches:        {len(test_loader)}")

    return train_loader, val_loader, test_loader, train_ds


def build_stats_loader(
    cfg: QueenPipelineConfig, train_ds: QueenAudioDataset
) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_queen_audio,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
    )


def configure_model(
    cfg: QueenPipelineConfig, mean: float, std: float
) -> ASTQueenClassifier:
    model = ASTQueenClassifier(
        model_name=cfg.model.model_name,
        dropout=cfg.model.dropout,
        mean=mean,
        std=std,
        do_normalize=cfg.model.do_normalize,
        target_sample_rate=cfg.audio.target_sample_rate,
    )

    if cfg.finetuning.freeze_strategy == "frozen":
        model.freeze_backbone()
    elif cfg.finetuning.freeze_strategy == "last_n":
        model.unfreeze_last_n_encoder_layers(cfg.finetuning.unfreeze_last_n)
    elif cfg.finetuning.freeze_strategy == "full":
        model.unfreeze_backbone()
    else:
        raise ValueError(
            f"Unknown freeze strategy: {cfg.finetuning.freeze_strategy}. Must be one of frozen, last_n, or full"
        )

    return model.to(DEVICE)


def build_optimizer(
    cfg: QueenPipelineConfig, model: ASTQueenClassifier
) -> torch.optim.Optimizer:
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    param_groups = []

    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": cfg.training.backbone_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            }
        )

    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": cfg.training.head_learning_rate,
                "weight_decay": cfg.training.weight_decay,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found")

    return torch.optim.AdamW(param_groups)


def train_one_epoch(
    cfg: QueenPipelineConfig,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    epoch_idx: int,
) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0
    epoch_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        batch_start = time.perf_counter()

        waveforms = batch["waveforms"]
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if cfg.training.use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip_norm
            )
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        if batch_idx % cfg.runtime.log_every_n_batches == 0 or batch_idx == len(loader):
            elapsed = time.perf_counter() - epoch_start
            batch_elapsed = time.perf_counter() - batch_start
            avg_loss = total_loss / total_count if total_count > 0 else float("nan")
            samples_per_sec = total_count / elapsed if elapsed > 0 else float("nan")

            print(
                f"[Train][Epoch {epoch_idx}] "
                f"batch {batch_idx}/{len(loader)} | "
                f"avg_loss={avg_loss:.4f} | "
                f"batch_time={batch_elapsed:.2f}s | "
                f"samples_seen={total_count} | "
                f"samples_per_sec={samples_per_sec:.2f} | "
                f"{format_gpu_mem()}"
            )

    epoch_loss = total_loss / total_count if total_count > 0 else math.nan
    epoch_time = time.perf_counter() - epoch_start
    print(
        f"[Train][Epoch {epoch_idx}] complete | "
        f"loss={epoch_loss:.4f} | "
        f"time={epoch_time:.2f}s"
    )
    return epoch_loss


@torch.no_grad()
def evaluate_model(
    cfg: QueenPipelineConfig,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    threshold: float,
    split_name: str = "Validation",
) -> dict:
    model.eval()

    total_loss = 0.0
    total_count = 0
    all_probs = []
    all_targets = []

    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):
        waveforms = batch["waveforms"]
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        logits = model(waveforms)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

        all_probs.append(probs.detach().cpu())
        all_targets.append(labels.detach().cpu())

        if batch_idx % cfg.runtime.log_every_n_batches == 0 or batch_idx == len(loader):
            print(
                f"[{split_name}] batch {batch_idx}/{len(loader)} | "
                f"{format_gpu_mem()}"
            )

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy()
    preds = (probs >= threshold).astype(np.float32)

    metrics = compute_classification_metrics(preds, targets)
    metrics["loss"] = total_loss / total_count if total_count > 0 else math.nan
    metrics["prob_mean"] = float(probs.mean())
    metrics["prob_std"] = float(probs.std())
    metrics["pred_pos_rate"] = float(preds.mean())
    metrics["eval_time_s"] = time.perf_counter() - start

    return metrics


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, loader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_probs = []
    all_targets = []

    for batch in loader:
        waveforms = batch["waveforms"]
        labels = batch["labels"]

        logits = model(waveforms)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(labels.detach().cpu())

    return torch.cat(all_probs).numpy(), torch.cat(all_targets).numpy()


def print_parameter_summary(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params:     {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")
    print(f"[Model] Frozen params:    {total - trainable:,}")


def train_queen_model(cfg: QueenPipelineConfig) -> dict:
    # Setup device and freeze strategy
    print(f"[Finetune] Device: {DEVICE}")
    print(f"[Finetune] Freeze strategy: {cfg.finetuning.freeze_strategy}")

    # Define project directory destinations
    # Models
    models_dir = cfg.paths.models_dir / cfg.project.name
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / cfg.paths.checkpoint_filename

    # Run metrics and dataset statistics (mean, std)
    metrics_path = models_dir / cfg.paths.metrics_filename
    stats_path = models_dir / cfg.paths.normalization_stats_filename

    # Construct dataloaders for training ds, validation ds, testing ds
    # Also setup training ds and settings for stats calculations
    train_loader, val_loader, test_loader, train_ds = build_loaders(cfg)
    stats_loader = build_stats_loader(cfg, train_ds)

    # Payload for run settings included with mean,std calculation
    stats_metadata = {
        "project_name": cfg.project.name,
        "subsample_fraction": cfg.split.subsample_fraction,
        "max_gap_days": cfg.labels.max_gap_days,
        "chunk_length_s": cfg.audio.chunk_length_s,
        "hop_length_s": cfg.audio.hop_length_s,
        "target_sample_rate": cfg.audio.target_sample_rate,
        "model_name": cfg.model.model_name,
    }

    # Compute mean and std for normalization of dataset
    # Note: Better performance was found with do_normalize off
    mean, std = compute_ast_stats(
        train_loader=stats_loader,
        model_name=cfg.model.model_name,
        target_sample_rate=cfg.audio.target_sample_rate,
        stats_json_path=stats_path,
        force_recompute=cfg.model.force_recompute_normalizations,
        stats_metadata=stats_metadata,
    )
    print(f"[Stats] mean={mean:.6f}, std={std:.6f}")

    # Set up feature extractor model
    # Use mean and std for normalization specific to UrBAN dataset
    # Do not use do_normalize=True without calculating the mean and
    # std without force_recompute_normalization=True at least one time
    # Must recompute whenever changing dataset
    model = configure_model(cfg=cfg, mean=mean, std=std)
    print_parameter_summary(model)
    optimizer = build_optimizer(cfg=cfg, model=model)

    # Determine if classes should be weighted based on frequency of occurance
    pos_weight = (
        compute_pos_weight(train_ds) if cfg.imbalance.use_class_weighting else None
    )
    print(
        f"[Train] pos_weight={None if pos_weight is None else float(pos_weight.item()):.4f}"
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # scaler = torch.cuda.amp.GradScaler() if USE_AMP else None
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.use_amp)

    # Initialize variables for tracking model performance during training loop
    best_val_bal_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    # Initialize variable for saving a description of the results of each epoch
    history = []

    for epoch in range(1, cfg.training.max_epochs + 1):
        print(f"\n[Finetune] Epoch {epoch}/{cfg.training.max_epochs}")

        # Train one epoch
        train_loss = train_one_epoch(
            cfg=cfg,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            epoch_idx=epoch,
        )

        # Evaluate results of epoch training
        val_metrics = evaluate_model(
            cfg=cfg,
            model=model,
            loader=val_loader,
            criterion=criterion,
            threshold=cfg.evaluation.default_threshold,
            split_name="Validation",
        )

        # Add epoch history to tracker
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
            }
        )

        # Print results from epoch training+validation
        print(
            f"[Finetune] train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_specificity={val_metrics['specificity']:.4f} | "
            f"prob_mean={val_metrics['prob_mean']:.4f} | "
            f"prob_std={val_metrics['prob_std']:.4f} | "
            f"pred_pos_rate={val_metrics['pred_pos_rate']:.4f} | "
            f"val_time={val_metrics['eval_time_s']:.2f}s"
        )

        # When current model performed better than best, replace best with current
        if val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save current best model in file to restore later
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_balanced_accuracy": best_val_bal_acc,
                    "config": {
                        "model_name": cfg.model.model_name,
                        "dropout": cfg.model.dropout,
                        "batch_size": cfg.training.batch_size,
                        "head_learning_rate": cfg.training.head_learning_rate,
                        "backbone_learning_rate": cfg.training.backbone_learning_rate,
                        "weight_decay": cfg.training.weight_decay,
                        "freeze_strategy": cfg.finetuning.freeze_strategy,
                        "unfreeze_last_n": cfg.finetuning.unfreeze_last_n,
                    },
                },
                best_model_path,
            )
            print(f"[Finetune] New best model saved to {best_model_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.training.patience:
            print(
                f"[Finetune] Early stopping after {cfg.training.patience} epochs without improvement."
            )
            break

    print(
        f"\n[Finetune] Best epoch: {best_epoch}, best val_bal_acc={best_val_bal_acc:.4f}"
    )

    # After training complete, load the best model for threshold search
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Determine the best threshold for the model by testing
    print("[Finetune] Collecting validation probabilities for threshold tuning...")
    val_probs, val_targets = predict_probabilities(model, val_loader)

    thresholds = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
    val_threshold_results = scan_thresholds(val_probs, val_targets, thresholds)

    # Print results for threshold search
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

    # Choose threshold that maximizes balanced accuracy
    best_threshold_result = choose_best_threshold_by_balanced_accuracy(
        val_threshold_results
    )
    best_threshold = best_threshold_result["threshold"]

    print(
        f"\n[Finetune] Best validation threshold: {best_threshold:.2f} | "
        f"bal_acc={best_threshold_result['balanced_accuracy']:.4f}"
    )

    print("[Finetune] Evaluating test set...")
    test_probs, test_targets = predict_probabilities(model, test_loader)
    test_metrics = metrics_at_threshold(test_probs, test_targets, best_threshold)

    print(
        f"[Test @ threshold={best_threshold:.2f}] "
        f"acc={test_metrics['accuracy']:.4f} | "
        f"bal_acc={test_metrics['balanced_accuracy']:.4f} | "
        f"f1={test_metrics['f1']:.4f} | "
        f"precision={test_metrics['precision']:.4f} | "
        f"recall={test_metrics['recall']:.4f} | "
        f"specificity={test_metrics['specificity']:.4f}"
    )

    payload = {
        "run_name": cfg.project.name,
        "description": cfg.project.description,
        "settings": cfg.to_dict(),
        "normalization": {
            "do_normalize": cfg.model.do_normalize,
            "force_recompute_normalizations": cfg.model.force_recompute_normalizations,
            "mean": mean,
            "std": std,
            "stats_path": str(stats_path),
        },
        "best_epoch": best_epoch,
        "best_val_balanced_accuracy": best_val_bal_acc,
        "selected_threshold": best_threshold,
        "validation_threshold_results": val_threshold_results,
        "history": history,
        "test_metrics": test_metrics,
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[Finetune] Metrics written to {metrics_path}")

    return payload
