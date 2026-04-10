from __future__ import annotations

from pathlib import Path
import pandas as pd
from hms_inference.audio_inspect_joiner import (
    attach_inspection_labels_2021,
    attach_inspection_labels_2022,
)
from hms_inference.config_loader import QueenPipelineConfig


def ensure_processed_dir(project_root: Path) -> Path:
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def build_dataset(cfg: QueenPipelineConfig) -> None:
    processed_dir = cfg.paths.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("[Build Dataset] Labeling 2021 Data:")
    labeled_2021 = attach_inspection_labels_2021(cfg)

    print("[Build Dataset] Labeling 2022 Data:")
    labeled_2022 = attach_inspection_labels_2022(cfg)

    print("[Build Dataset] Saving labeled data as parquet:")
    labeled_2021.to_parquet(
        processed_dir / "urban_labeled_data_2021.parquet", index=False
    )
    labeled_2022.to_parquet(
        processed_dir / "urban_labeled_data_2022.parquet", index=False
    )

    print("\n[Build Dataset] Quick summary:")
    print("2021 queen_present counts:")
    print(labeled_2021["queen_present"].value_counts(dropna=False))

    print("\n2022 queen_present counts:")
    print(labeled_2022["queen_present"].value_counts(dropna=False))

    if "hive_state" in labeled_2022.columns:
        print("\n2022 hive_state counts:")
        print(labeled_2022["hive_state"].value_counts(dropna=False))

    if "varroa_high" in labeled_2022.columns:
        print("\n2022 varroa_high counts:")
        print(labeled_2022["varroa_high"].value_counts(dropna=False))

    print("\n[Build Dataset] Done.")
    print(f"[Build Dataset] Outputs written to: {processed_dir}")
