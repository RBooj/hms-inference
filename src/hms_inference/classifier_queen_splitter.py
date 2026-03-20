from __future__ import annotations

from pathlib import Path
from itertools import combinations

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
    df_2021 = pd.read_parquet(processed_dir / "urban_labeled_data_2021.parquet")
    df_2022 = pd.read_parquet(processed_dir / "urban_labeled_data_2022.parquet")

    df = pd.concat([df_2021, df_2022], ignore_index=True)
    return df


def build_queen_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Drop rows without queen data
    out = out[out["queen_present"].notna()].copy()

    # Drop columns unrelated to queen detection task
    out = out[QUEEN_TASK_COLUMNS].copy()

    return out


def per_hive_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("hive_id")["queen_present"]
        .agg(
            total="count",
            positives="sum",
        )
        .reset_index()
    )
    stats["negatives"] = stats["total"] - stats["positives"]
    stats["pos_rate"] = stats["positives"] / stats["total"]
    return stats.sort_values("hive_id").reset_index(drop=True)


def split_stats(hive_stats: pd.DataFrame, hive_ids: list[int]) -> dict:
    sub = hive_stats[hive_stats["hive_id"].isin(hive_ids)].copy()
    total = int(sub["total"].sum())
    positives = int(sub["positives"].sum())
    negatives = int(sub["negatives"].sum())
    pos_rate = positives / total if total > 0 else float("nan")
    return {
        "hives": list(hive_ids),
        "n_hives": len(hive_ids),
        "total": total,
        "positives": positives,
        "negatives": negatives,
        "pos_rate": pos_rate,
    }


def score_split(train: dict, val: dict, test: dict, global_pos_rate: float) -> float:
    # Lower is better.
    # Main goal: similar class balance across splits.
    # Small extra penalty for wildly uneven dataset sizes.
    balance_penalty = (
        abs(train["pos_rate"] - global_pos_rate)
        + abs(val["pos_rate"] - global_pos_rate)
        + abs(test["pos_rate"] - global_pos_rate)
    )

    size_penalty = (
        abs(train["n_hives"] - 7) * 0.01
        + abs(val["n_hives"] - 2) * 0.01
        + abs(test["n_hives"] - 1) * 0.01
    )

    return balance_penalty + size_penalty


def has_both_classes(stats: dict) -> bool:
    return stats["positives"] > 0 and stats["negatives"] > 0


def find_best_split(hive_stats: pd.DataFrame) -> tuple[dict, dict, dict, float]:
    hive_ids = hive_stats["hive_id"].tolist()
    global_pos_rate = hive_stats["positives"].sum() / hive_stats["total"].sum()

    best = None
    best_score = float("inf")

    # Choose 1 test hive, 2 val hives, remaining 7 train hives
    for test_hives in combinations(hive_ids, 1):
        remaining_after_test = [h for h in hive_ids if h not in test_hives]

        for val_hives in combinations(remaining_after_test, 2):
            train_hives = [h for h in remaining_after_test if h not in val_hives]

            train = split_stats(hive_stats, train_hives)
            val = split_stats(hive_stats, list(val_hives))
            test = split_stats(hive_stats, list(test_hives))

            if not (
                has_both_classes(train)
                and has_both_classes(val)
                and has_both_classes(test)
            ):
                continue

            score = score_split(train, val, test, global_pos_rate)

            if score < best_score:
                best_score = score
                best = (train, val, test, global_pos_rate)

    if best is None:
        raise RuntimeError("No valid split found with both classes in every split.")

    return best[0], best[1], best[2], best[3]


def filter_by_hives(df: pd.DataFrame, hive_ids: list[int]) -> pd.DataFrame:
    return df[df["hive_id"].isin(hive_ids)].copy()


def print_split_summary(name: str, stats: dict) -> None:
    print(f"\n{name} split")
    print("-" * (len(name) + 6))
    print(f"hives:     {stats['hives']}")
    print(f"rows:      {stats['total']}")
    print(f"positives: {stats['positives']}")
    print(f"negatives: {stats['negatives']}")
    print(f"pos_rate:  {stats['pos_rate']:.4f}")


def print_dataframe_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n{name} dataframe summary")
    print("-" * (len(name) + 18))
    print(f"rows:  {len(df)}")
    print(f"hives: {df['hive_id'].nunique()}")
    print(df["queen_present"].value_counts(dropna=False))


def create_queen_splits() -> None:
    project_root = Path.cwd()
    processed_dir = project_root / "data" / "processed"
    splits_dir = project_root / "data" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("[Queen Split] Loading labeled chunk parquet files...")
    labeled_df = load_labeled_chunks(processed_dir)

    queen_df = build_queen_dataset(labeled_df)
    hive_stats = per_hive_stats(queen_df)

    print("\n[Queen Split] Per-hive stats")
    print(hive_stats.to_string(index=False))

    train_stats, val_stats, test_stats, global_pos_rate = find_best_split(hive_stats)

    print(f"\n[Queen Split] Global positive rate: {global_pos_rate:.4f}")
    print_split_summary("Train", train_stats)
    print_split_summary("Validation", val_stats)
    print_split_summary("Test", test_stats)

    queen_train = filter_by_hives(queen_df, train_stats["hives"])
    queen_val = filter_by_hives(queen_df, val_stats["hives"])
    queen_test = filter_by_hives(queen_df, test_stats["hives"])

    queen_train.to_parquet(splits_dir / "queen_train.parquet", index=False)
    queen_val.to_parquet(splits_dir / "queen_val.parquet", index=False)
    queen_test.to_parquet(splits_dir / "queen_test.parquet", index=False)

    print_dataframe_summary("Train", queen_train)
    print_dataframe_summary("Validation", queen_val)
    print_dataframe_summary("Test", queen_test)

    print("\nSaved files:")
    print(splits_dir / "queen_train.parquet")
    print(splits_dir / "queen_val.parquet")
    print(splits_dir / "queen_test.parquet")
