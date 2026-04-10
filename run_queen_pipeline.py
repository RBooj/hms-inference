from __future__ import annotations

import argparse
import json
from pathlib import Path

from hms_inference.config_loader import load_config
from hms_inference.dataset_builder import build_dataset
from hms_inference.classifier_queen_splitter import create_queen_splits
from hms_inference.classifier_queen_finetune import train_queen_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_dir = cfg.paths.models_dir / cfg.project.name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = run_dir / "run_config_snapshot.json"

    with config_snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print(f"[Runner] Loaded config from: {args.config}")
    print(f"[Runner] Saved config snapshot to: {config_snapshot_path}")

    print("[Runner] Step 1/3: build dataset")
    build_dataset(cfg)

    print("[Runner] Step 2/3: create queen splits")
    create_queen_splits(cfg)

    print("[Runner] Step 3/3: train queen model")
    train_queen_model(cfg)


if __name__ == "__main__":
    main()
