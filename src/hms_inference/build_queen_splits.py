from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

RANDOM_SEED = 42

QUEEN_TASK_COLUMNS = [
    "dataset_year",
    "wav_path",
    "hive_id",
    "recording_start_dt",
    "chunk_idx",
    "chunk_start_s",
    "chunk_end_s",
    "chunk_start_dt",
    "chunk_end_dt",
    "inspection_date",
    "days_since_inspection",
    "queen_present",
]


def load_labeled_chunks(processed_dir: Path) -> pd.DataFrame:
    df_2021 = pd.read_parquet(processed_dir / "urban_chunks_2021_labeled.parquet")
    df_2022 = pd.read_parquet(processed_dir / "urban_chunks_2022_labeled.parquet")

    df = pd.concat([df_2021, df_2022], ignore_index=True)
    return df


def build_queen_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Drop rows without queen data
    out = out[out["queen_present"].notna()].copy()

    # Drop columns unrelated to queen detection task
    out = out[QUEEN_TASK_COLUMNS].copy()

    return out


def split_hives(
    hive_ids: list[int], *, seed: int = RANDOM_SEED
) -> tuple[list[int], list[int], list[int]]:
    """
    Dataset is split by hive ID
    Creates three lists containing the hive ids used for:
        1. training (~70%)
        2. validation (~15%)
        3. testing (~15%)
    Add random chance to shuffle total number of hives in each task
    for variation
    """
    hive_ids = list(hive_ids)
    rng = random.Random(seed)
    rng.shuffle(hive_ids)

    n = len(hive_ids)

    n_train = max(1, round(n * 0.7))
    n_val = max(1, round(n * 0.15))
    n_test = n - n_train - n_val

    # Ensure at least 1 hive in test
    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_hives = hive_ids[:n_train]
    val_hives = hive_ids[n_train : n_train + n_val]
    test_hives = hive_ids[n_train + n_val :]

    return train_hives, val_hives, test_hives


def filter_by_hive_id(df: pd.DataFrame, hive_ids: list[int]) -> pd.DataFrame:
    return df[df["hive_id"].isin(hive_ids)].copy()


def print_split_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name} summary")
    print("-" * (len(name) + 8))
    print(f"rows:  {len(df)}")
    print(f"hives: {df['hive_id'].nunique()}")
    print("queen_present counts:")
    print(df["queen_present"].value_counts(dropna=False))


def main() -> None:
    project_root = Path.cwd()
    processed_dir = project_root / "data" / "processed"
    splits_dir = project_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("[Queen Split] Loading labeled chunk parquet files...")
    labeled_df = load_labeled_chunks(processed_dir)

    print(f"[Queen Split] Combined labeled rows: {len(labeled_df)}")

    queen_df = build_queen_dataset(labeled_df)

    print(f"[Queen Split] Rows with queen labels: {len(queen_df)}")
    print(f"[Queen Split] Unique hives: {queen_df['hive_id'].nunique()}")

    unique_hives = sorted(queen_df["hive_id"].dropna().unique().tolist())
    train_hives, val_hives, test_hives = split_hives(unique_hives, seed=RANDOM_SEED)

    print("\n[Queen Split] Hive assignments")
    print(f"train hives ({len(train_hives)}): {train_hives}")
    print(f"val hives   ({len(val_hives)}): {val_hives}")
    print(f"test hives  ({len(test_hives)}): {test_hives}")

    queen_train = filter_by_hive_id(queen_df, train_hives)
    queen_val = filter_by_hive_id(queen_df, val_hives)
    queen_test = filter_by_hive_id(queen_df, test_hives)

    queen_train.to_parquet(splits_dir / "queen_train.parquet", index=False)
    queen_val.to_parquet(splits_dir / "queen_val.parquet", index=False)
    queen_test.to_parquet(splits_dir / "queen_test.parquet", index=False)

    print_split_summary("Train", queen_train)
    print_split_summary("Validation", queen_val)
    print_split_summary("Test", queen_test)

    print("\n[Queen Split] Saved files:")
    print(splits_dir / "queen_train.parquet")
    print(splits_dir / "queen_val.parquet")
    print(splits_dir / "queen_test.parquet")


if __name__ == "__main__":
    main()
