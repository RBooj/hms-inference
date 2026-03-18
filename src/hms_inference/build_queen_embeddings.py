from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from hms_inference.audio_io import load_audio_mono_16k, slice_waveform_chunk
from hms_inference.ast_embedder import ASTEmbedder

TEST_SPLIT_NAME = "queen_train"
TEST_LIMIT_ROWS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_label(value) -> int:
    """
    Convert queen_present boolean label into integer class.
    True -> 1
    False -> 0
    """
    if pd.isna(value):
        raise ValueError("Found missing queen_present label in embedding builder.")
    return int(bool(value))


def build_embeddings_for_split(
    split_df: pd.DataFrame,
    embedder: ASTEmbedder,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Returns:
        meta_df: metadata aligned row-for-row with embeddings
        X: embedding matrix, shape [N, D]
        y: labels, shape [N]
    """

    meta_rows: list[dict] = []
    embedding_rows: list[np.ndarray] = []
    label_rows: list[int] = []

    # group by wav_path so each wav is loaded only once
    grouped = split_df.groupby("wav_path", sort=False)

    total_groups = grouped.ngroups
    for group_idx, (wav_path, group_df) in enumerate(grouped, start=1):
        print(f"[Embed] Loading wav {group_idx}/{total_groups}: {wav_path}")

        waveform = load_audio_mono_16k(wav_path)

        # Preserve row order within the grouped frame
        for _, row in group_df.iterrows():
            chunk = slice_waveform_chunk(
                waveform,
                start_s=float(row["chunk_start_s"]),
                end_s=float(row["chunk_end_s"]),
                sample_rate=16000,
            )

            # good sanity check for 10-second chunks
            if chunk.numel() != 160000:
                print(
                    "[Embed] Warning: chunk length is not exactly 160000 samples. "
                    f"Got {chunk.numel()} for wav={wav_path}, chunk_idx={row['chunk_idx']}"
                )
            result = embedder.embed(chunk)
            emb = result.embedding

            # convert embedding tensor -> 1D numpy array
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            else:
                emb = np.asarray(emb)

            emb = np.asarray(emb).squeeze()

            if emb.ndim != 1:
                raise ValueError(
                    f"Expected 1D embedding vector, got shape {emb.shape} "
                    f"for wav={wav_path}, chunk_idx={row['chunk_idx']}"
                )

            label = normalize_label(row["queen_present"])

            meta_rows.append(
                {
                    "dataset_year": row["dataset_year"],
                    "wav_path": row["wav_path"],
                    "hive_id": row["hive_id"],
                    "recording_start_dt": row["recording_start_dt"],
                    "chunk_idx": row["chunk_idx"],
                    "chunk_start_s": row["chunk_start_s"],
                    "chunk_end_s": row["chunk_end_s"],
                    "chunk_start_dt": row["chunk_start_dt"],
                    "chunk_end_dt": row["chunk_end_dt"],
                    "inspection_date": row["inspection_date"],
                    "days_since_inspection": row["days_since_inspection"],
                    "queen_present": row["queen_present"],
                }
            )
            embedding_rows.append(emb)
            label_rows.append(label)

    meta_df = pd.DataFrame(meta_rows)
    X = np.stack(embedding_rows, axis=0)
    y = np.asarray(label_rows, dtype=np.int64)

    return meta_df, X, y


def main() -> None:
    project_root = Path.cwd()
    splits_dir = project_root / "data" / "splits"
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    split_path = splits_dir / f"{TEST_SPLIT_NAME}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(f"Split parquet not found: {split_path}")

    print(f"[Embed] Loading split: {split_path}")
    df = pd.read_parquet(split_path)

    # smoke test subset
    df = df.head(TEST_LIMIT_ROWS).copy()

    print(f"[Embed] Rows selected for test: {len(df)}")
    print(f"[Embed] Unique wavs in test subset: {df['wav_path'].nunique()}")
    print(f"[Embed] Label counts:\n{df['queen_present'].value_counts(dropna=False)}")

    print(f"[Embed] Initializing AST embedder on device={DEVICE}")
    embedder = ASTEmbedder(device=DEVICE)

    meta_df, X, y = build_embeddings_for_split(df, embedder)

    print(f"[Embed] Metadata rows: {len(meta_df)}")
    print(f"[Embed] Embedding matrix shape: {X.shape}")
    print(f"[Embed] Label vector shape: {y.shape}")
    print(f"[Embed] Label counts (0/1): {np.bincount(y)}")

    # save outputs
    meta_out = embeddings_dir / f"{TEST_SPLIT_NAME}_test_meta.parquet"
    x_out = embeddings_dir / f"{TEST_SPLIT_NAME}_test_embeddings.npy"
    y_out = embeddings_dir / f"{TEST_SPLIT_NAME}_test_labels.npy"

    meta_df.to_parquet(meta_out, index=False)
    np.save(x_out, X)
    np.save(y_out, y)

    print("\n[Embed] Saved test outputs:")
    print(meta_out)
    print(x_out)
    print(y_out)


if __name__ == "__main__":
    main()
