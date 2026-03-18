"""
Load and normalize 2021 inspections
Load and normalize 2022 inspections
Discover all wav files
Build chunk labels
Join inspections into chunks
Save output as parquet
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from hms_inference.audio_discovery import find_wavs, AUDIO_ROOT_2021, AUDIO_ROOT_2022
from hms_inference.audio_builder import build_chunk_index
from hms_inference.inspections_loader import (
    load_inspections_2021,
    load_inspections_2022,
)
from hms_inference.label_join import (
    attach_inspection_labels_2021,
    attach_inspection_labels_2022,
)

MAX_GAP_DAYS = 21


def ensure_processed_dir(project_root: Path) -> Path:
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def main() -> None:
    project_root = Path.cwd()
    processed_dir = ensure_processed_dir(project_root)

    print("\n[Build Datasset] Loading and normalizing inspections...")

    inspections_2021 = load_inspections_2021(project_root)
    inspections_2022 = load_inspections_2022(project_root)

    print(f"[Build] 2021 inspections: {len(inspections_2021)} rows")
    print(f"[Build] 2022 inspections: {len(inspections_2022)} rows")

    print("\n[Build] Discovering audio...")
    wavs_2021 = find_wavs(AUDIO_ROOT_2021)
    wavs_2022 = find_wavs(AUDIO_ROOT_2022)

    print(f"[Build] 2021 WAVs found: {len(wavs_2021)}")
    print(f"[Build] 2022 WAVs found: {len(wavs_2022)}")

    print("\n[Build] Building audio chunk indices...")
    chunks_2021 = build_chunk_index(wavs_2021, dataset_year=2021)
    chunks_2022 = build_chunk_index(wavs_2022, dataset_year=2022)

    print(f"[Build] 2021 audio chunk rows: {len(chunks_2021)}")
    print(f"[Build] 2022 audio chunk rows: {len(chunks_2022)}")

    print("\n[Build] Attaching labels...")
    labeled_2021 = attach_inspection_labels_2021(
        chunks_2021,
        inspections_2021,
        max_gap_days=MAX_GAP_DAYS,
    )
    labeled_2022 = attach_inspection_labels_2022(
        chunks_2022,
        inspections_2022,
        max_gap_days=MAX_GAP_DAYS,
    )

    print(f"[Build] 2021 labeled chunk rows: {len(labeled_2021)}")
    print(f"[Build] 2022 labeled chunk rows: {len(labeled_2022)}")

    print("\n[Build] Saving parquet files...")
    inspections_2021.to_parquet(
        processed_dir / "urban_inspections_2021_normalized.parquet",
        index=False,
    )
    inspections_2022.to_parquet(
        processed_dir / "urban_inspections_2022_normalized.parquet",
        index=False,
    )
    labeled_2021.to_parquet(
        processed_dir / "urban_chunks_2021_labeled.parquet",
        index=False,
    )
    labeled_2022.to_parquet(
        processed_dir / "urban_chunks_2022_labeled.parquet",
        index=False,
    )

    print("\n[Build] Quick summary:")
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

    print("\n[Build] Done.")
    print(f"[Build] Outputs written to: {processed_dir}")


if __name__ == "__main__":
    main()
