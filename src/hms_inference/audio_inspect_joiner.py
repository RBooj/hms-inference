from __future__ import annotations
import pandas as pd
from pathlib import Path

from hms_inference.inspections_loader import (
    load_inspections_2021,
    load_inspections_2022,
)
from hms_inference.audio_loader import discover_wav_files, build_chunk_df
from hms_inference.config_loader import QueenPipelineConfig

# MAX_GAP_DAYS = 1  # Be > 0


def attach_inspection_labels_2022(cfg: QueenPipelineConfig) -> pd.DataFrame:
    """
    Extract annotations from inspections_2022.csv spreadsheet
    Extract audio file metadata from filenames
    split audio files into chunks with the closest inspection record
    attach annotations to audio chunks
    """
    project_root = cfg.project.project_root
    audio_root = project_root / "data" / "UrBAN" / "data" / "audio" / "beehives_2022"
    tolerance = pd.Timedelta(days=cfg.labels.max_gap_days)

    wavs_2022 = discover_wav_files(audio_root)
    print(f"[Attach Labels 2022] Discovered {len(wavs_2022)} wavs in 2022 audio folder")

    chunks_2022 = build_chunk_df(
        wavs_2022,
        2022,
        chunk_length_s=cfg.audio.chunk_length_s,
        hop_length_s=cfg.audio.hop_length_s,
    )
    print(f"[Attach Labels 2022] Built {len(chunks_2022)} chunks.")

    inspections_2022 = load_inspections_2022(project_root)
    print(f"[Attach Labels 2022] Loaded {len(inspections_2022)} inspection records.")

    chunks_2022 = chunks_2022.sort_values(["chunk_start_dt", "hive_id"]).reset_index(
        drop=True
    )
    inspections_2022 = inspections_2022.sort_values(
        ["inspection_date", "hive_id"]
    ).reset_index(drop=True)

    labeled = pd.merge_asof(
        chunks_2022,
        inspections_2022[
            [
                "hive_id",
                "inspection_date",
                "frames_of_bees",
                "hive_state",
                "queen_present",
                "varroa_high",
            ]
        ],
        by="hive_id",
        left_on="chunk_start_dt",
        right_on="inspection_date",
        direction="backward",
        tolerance=tolerance,
    )

    labeled["days_since_inspection"] = (
        labeled["chunk_start_dt"] - labeled["inspection_date"]
    ).dt.total_seconds() / 86400.0

    print("[Attach Labels 2022] Labeled Data Stats:")
    print("\nQueen distribution:")
    print(labeled["queen_present"].value_counts(dropna=False))

    print("\nHive state distribution:")
    print(labeled["hive_state"].value_counts(dropna=False))

    print("\nVarroa high stats:")
    print(labeled["varroa_high"].describe())

    print("\nDays since inspection (describe):")
    print(labeled["days_since_inspection"].describe())

    return labeled


def attach_inspection_labels_2021(cfg: QueenPipelineConfig) -> pd.DataFrame:
    """
    Extract annotations from inspections_2021.csv spreadsheet
    Extract audio file metadata from filenames
    split audio files into chunks with the closest inspection record
    attach annotations to audio chunks
    """
    project_root = cfg.project.project_root
    audio_root = project_root / "data" / "UrBAN" / "data" / "audio" / "beehives_2021"
    tolerance = pd.Timedelta(days=cfg.labels.max_gap_days)

    wavs_2021 = discover_wav_files(audio_root)
    print(f"[Attach Labels 2021] Discovered {len(wavs_2021)} wavs in 2021 audio folder")

    chunks_2021 = build_chunk_df(
        wavs_2021,
        2021,
        chunk_length_s=cfg.audio.chunk_length_s,
        hop_length_s=cfg.audio.hop_length_s,
    )
    print(f"[Attach Labels 2021] Built {len(chunks_2021)} chunks.")

    inspections_2021 = load_inspections_2021(project_root)
    print(f"[Attach Labels 2021] Loaded {len(inspections_2021)} inspection records.")

    chunks_2021 = chunks_2021.sort_values(["chunk_start_dt", "hive_id"]).reset_index(
        drop=True
    )
    inspections_2021 = inspections_2021.sort_values(
        ["inspection_date", "hive_id"]
    ).reset_index(drop=True)

    labeled = pd.merge_asof(
        chunks_2021,
        inspections_2021[["hive_id", "inspection_date", "queen_present", "fob_total"]],
        by="hive_id",
        left_on="chunk_start_dt",
        right_on="inspection_date",
        direction="backward",
        tolerance=tolerance,
    )

    labeled["days_since_inspection"] = (
        labeled["chunk_start_dt"] - labeled["inspection_date"]
    ).dt.total_seconds() / 86400.0

    print("[Attach Labels 2021] Labeled Data Stats:")
    print("\nqueen present counts (including NaNs): ")
    print(labeled["queen_present"].value_counts(dropna=False))

    print("\nDays since inspection (describe): ")
    print(labeled["days_since_inspection"].describe())

    print("\nFrames of Bees information: ")
    print(labeled["fob_total"].describe())

    return labeled
