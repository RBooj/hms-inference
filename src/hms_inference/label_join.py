from __future__ import annotations

from pathlib import Path
import pandas as pd

# import functions from project
from hms_inference.audio_discovery import find_wavs, AUDIO_ROOT_2021, AUDIO_ROOT_2022
from hms_inference.audio_builder import build_chunk_index
from hms_inference.inspections_loader import load_inspections_2021, load_inspections_2022

def attach_inspection_labels_2022(chunks: pd.DataFrame, inspections: pd.DataFrame, *, max_gap_days: int | None = None) -> pd.DataFrame:
    """
    Extract annotations from inspections_2022.csv spreadsheet using load_inspections_2021
    Correlate the chunked audio files with the annotations so that each chunk is labeled with hive information
    Attach labels to each chunk using:
        - matching hive_id
        - most recent inspection date at or before the start time of the audio chunk

    Do not label chunks that are more than max_gap_days from the last inspection
    """

    chunks = chunks.copy()
    inspections = inspections.copy()

    chunks["chunk_start_dt"] = pd.to_datetime(chunks["chunk_start_dt"], utc=True)
    inspections["inspection_date"] = pd.to_datetime(inspections["inspection_date"], utc=True)

    # short by hive_id then date
    chunks = chunks.sort_values(["chunk_start_dt", "hive_id"]).reset_index(drop=True)
    inspections = inspections.sort_values(["inspection_date", "hive_id"]).reset_index(drop=True)

    tolerance = None
    if max_gap_days is not None:
        tolerance = pd.Timedelta(days=max_gap_days)

    labeled = pd.merge_asof(
            chunks,
            inspections[["hive_id", "inspection_date", "frames_of_bees", "hive_state", "queen_present", "varroa_high"]],
            by="hive_id",
            left_on="chunk_start_dt",
            right_on="inspection_date",
            direction="backward",
            tolerance=tolerance,
            )

    # include a column stating the time distance between the chunk and its matched inspection
    labeled["days_since_inspection"] = (
        labeled["chunk_start_dt"] - labeled["inspection_date"]
    ).dt.total_seconds() / 86400.0

    return labeled

def attach_inspection_labels_2021(
    chunks: pd.DataFrame, inspections: pd.DataFrame, *, max_gap_days: int | None = None
) -> pd.DataFrame:
    """
    Extract annotations from inspections_2021.csv spreadsheet using load_inspections_2021
    Correlate the chunked audio files with the annotations so that each chunk is labeled with hive information
    Attach labels to each chunk using:
        - matching hive_id
        - most recent inspection date at or before the start time of the audio chunk

    Do not label chunks that are more than max_gap_days from the last inspection
    """

    # datetime types are used for time-aware joins
    chunks = chunks.copy()
    inspections = inspections.copy()

    chunks["chunk_start_dt"] = pd.to_datetime(chunks["chunk_start_dt"], utc=True)
    inspections["inspection_date"] = pd.to_datetime(inspections["inspection_date"], utc=True)

    # sort by hive_id, then date
    chunks = chunks.sort_values(["chunk_start_dt", "hive_id"]).reset_index(drop=True)
    inspections = inspections.sort_values(["inspection_date", "hive_id"]).reset_index(
        drop=True
    )

    # Implement tolerance window from max_gap_days
    tolerance = None
    if max_gap_days is not None:
        tolerance = pd.Timedelta(days=max_gap_days)

    labeled = pd.merge_asof(
        chunks,
        inspections[["hive_id", "inspection_date", "queen_present", "fob_total"]],
        by="hive_id",
        left_on="chunk_start_dt",
        right_on="inspection_date",
        direction="backward",
        tolerance=tolerance,
    )

    # include a column stating the time distance between the chunk and its matched inspection
    labeled["days_since_inspection"] = (
        labeled["chunk_start_dt"] - labeled["inspection_date"]
    ).dt.total_seconds() / 86400.0

    # 2021 data does not have sufficient data and/or granular enough application of
    # "frames of bees" measurement - leaving this identifier out of 2021 labeled data
    # labeled["strength_class"] = labeled["fob_total"].apply(derive_strength_class)

    return labeled


# def derive_strength_class(fob_total: float | None) -> str | None:
#     """
#     Arbitrarily guessed delimeters here.
#     Less than 10 FOB: weak
#     Less than 20 FOB: medium
#     20 or more FOB:   strong
#     """
#     if pd.isna(fob_total):
#         return None
#     if fob_total < 10:
#         return "weak"
#     elif fob_total < 20:
#         return "medium"
#     else:
#         return "strong"


if __name__ == "__main__":
    print("\n2021 Chunked Data ------------------------------------------------------")
    project_root = Path.cwd()

    # build chunk index for 2021
    wavs_2021 = find_wavs(AUDIO_ROOT_2021)
    print("2021 wavs found: ", len(wavs_2021))

    chunks = build_chunk_index(wavs_2021, dataset_year=2021)
    print("chunk rows: ", len(chunks), "(expected: ", len(wavs_2021) * 5, ")")

    # load inspections
    inspections = load_inspections_2021(project_root)
    print(
        "inspection rows: ",
        len(inspections),
        "hives: ",
        inspections["hive_id"].nunique(),
    )

    # attach labels. Leave tolerance blank for this test
    labeled = attach_inspection_labels_2021(chunks, inspections, max_gap_days=21)

    # display debug information
    print("\nLabeled head (important columns): ")
    cols = [
        "hive_id",
        "chunk_start_dt",
        "wav_path",
        "chunk_idx",
        "inspection_date",
        "queen_present",
        "days_since_inspection",
    ]
    print(labeled[cols].head(10).to_string(index=False))

    print("\nqueen present counts (including NaNs): ")
    print(labeled["queen_present"].value_counts(dropna=False))

    print("\nDays since inspection (describe): ")
    print(labeled["days_since_inspection"].describe())

    print("chunks hive_id NA:", chunks["hive_id"].isna().sum())
    print("inspections hive_id NA:", inspections["hive_id"].isna().sum())

    print("\nFrames of Bees information: ")
    print(labeled["fob_total"].describe())
    # print(labeled["strength_class"].value_counts(dropna=False))

    print("\n2022 Chunked Data ------------------------------------------------------")
    wavs_2022 = find_wavs(AUDIO_ROOT_2022)
    print("2022 wavs found: ", len(wavs_2022))

    chunks_2022 = build_chunk_index(wavs_2022, dataset_year=2022)
    print("chunk rows: ", len(chunks_2022), "(expected: ", len(wavs_2022) * 5, ")")

    inspections_2022 = load_inspections_2022(project_root)
    print(
            "inspection rows: ",
            len(inspections_2022),
            "hives: ",
            inspections["hive_id"].nunique()
            )

    labeled_2022 = attach_inspection_labels_2022(chunks_2022, inspections_2022, max_gap_days=21)

    important_cols = [
            "hive_id",
            "chunk_start_dt",
            "wav_path",
            "chunk_idx",
            "inspection_date",
            "queen_present",
            "hive_state",
            "frames_of_bees",
            "varroa_high",
            "days_since_inspection",
            ]

    print("\nLabeled head (important columns):")
    print(labeled_2022[important_cols].head(10).to_string(index=False))

    print("\nQueen distribution:")
    print(labeled_2022["queen_present"].value_counts(dropna=False))

    print("\nHive state distribution:")
    print(labeled_2022["hive_state"].value_counts(dropna=False))

    print("\nVarroa high stats:")
    print(labeled_2022["varroa_high"].describe())

    print("\nDays since inspection (describe):")
    print(labeled_2022["days_since_inspection"].describe())

    labeled_2022.to_csv("labeled_2022_data.csv")
