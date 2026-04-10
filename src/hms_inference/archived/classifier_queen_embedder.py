from __future__ import annotations
from pathlib import Path

import gc
import json
import numpy as np
import pandas as pd
import torch
import torchaudio

from hms_inference.ast_embedder import ASTEmbedder

SPLITS = ["queen_val", "queen_test", "queen_train"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONSECUTIVE_FILES = 500
MAX_ROWS_PER_SHARD = 100000
EMBED_BATCH_SIZE = 180


def slice_waveform_chunk(
    waveform_16k: torch.Tensor, start_s: float, end_s: float, sample_rate: int = 16000
) -> torch.Tensor:
    start_idx = int(round(start_s * sample_rate))
    end_idx = int(round(end_s * sample_rate))

    chunk = waveform_16k[start_idx:end_idx]

    if chunk.numel() == 0:
        raise ValueError(
            f"Chunkslice is empty: start_s={start_s}, end_s={end_s}, "
            f"start_idx={start_idx}, end_idx={end_idx}, waveform_len={waveform_16k.numel()}"
        )

    return chunk


def load_audio_mono_16k(path: str) -> torch.Tensor:
    """
    Load audio file as single-channel wav sampled at 16kHz
    Return waveform shape [num_samples] (float32, -1..1)
    """
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.squeeze(0).to(torch.float32)
    return wav


def normalize_label(value) -> int:
    """
    Convert queen_present boolean label into integer class.
    True -> 1
    False -> 0
    """
    if pd.isna(value):
        raise ValueError("Found missing queen_present label in embedding builder.")
    return int(bool(value))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def load_progress(progress_path: Path, total_wav_groups: int) -> dict:
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)
    else:
        progress = {
            "complete": False,
            "next_wav_group_idx": 0,
            "next_shard_idx": 0,
            "processed_wav_groups": 0,
            "total_wav_groups": total_wav_groups,
        }
    return progress


def flush_shard(
    split_output_dir: Path,
    shard_idx: int,
    meta_rows: list[dict],
    embedding_rows: list[np.ndarray],
    label_rows: list[int],
) -> None:
    if not meta_rows:
        return

    meta_df = pd.DataFrame(meta_rows)
    X = np.stack(embedding_rows, axis=0)
    y = np.asarray(label_rows, dtype=np.int64)

    base_name = f"part_{shard_idx:05d}"

    meta_path = split_output_dir / f"{base_name}_meta.parquet"
    emb_path = split_output_dir / f"{base_name}_embeddings.npy"
    label_path = split_output_dir / f"{base_name}_labels.npy"

    meta_df.to_parquet(meta_path, index=False)
    np.save(emb_path, X)
    np.save(label_path, y)

    print(
        f"[Embed] Saved shard {base_name}: "
        f"rows={len(meta_df)}, embeddings_shape={X.shape}, labels_shape={y.shape}"
    )


def clear_buffers(
    meta_rows: list[dict],
    embedding_rows: list[np.ndarray],
    label_rows: list[int],
) -> None:
    meta_rows.clear()
    embedding_rows.clear()
    label_rows.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def process_split(
    split: str,
    split_path: Path,
    split_output_dir: Path,
    embedder: ASTEmbedder,
) -> None:
    if not split_path.exists():
        raise FileNotFoundError(f"Split parquet not found: {split_path}")

    print(f"\n[Embed] Loading split: {split_path}")
    df = pd.read_parquet(split_path)

    # Deterministic ordering for resumability
    df = df.sort_values(["wav_path", "chunk_idx"]).reset_index(drop=True)

    wavs_in_split = df["wav_path"].nunique()
    print(f"[Embed] Rows in split: {len(df)}")
    print(f"[Embed] Unique wavs in split: {wavs_in_split}")
    print(f"[Embed] Label counts:\n{df['queen_present'].value_counts(dropna=False)}")

    split_output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = split_output_dir / "progress.json"

    progress = load_progress(progress_path, wavs_in_split)

    if progress.get("complete", False):
        print(f"[Embed] Split {split} already complete. Skipping.")
        return

    start_wav_group_idx = int(progress.get("next_wav_group_idx", 0))
    shard_idx = int(progress.get("next_shard_idx", 0))

    print(
        f"[Embed] Resuming split {split} from wav group {start_wav_group_idx}/{wavs_in_split}, "
        f"next shard index={shard_idx}"
    )

    meta_rows: list[dict] = []
    embedding_rows: list[np.ndarray] = []
    label_rows: list[int] = []

    wavs_in_current_shard = 0

    grouped = df.groupby("wav_path", sort=True)

    for wav_group_idx, (wav_path, group_df) in enumerate(grouped):
        if wav_group_idx < start_wav_group_idx:
            continue

        print(
            f"[Embed] {split}: Loading wav {wav_group_idx + 1}/{wavs_in_split}: {wav_path}"
        )

        waveform = load_audio_mono_16k(wav_path)
        group_df = group_df.sort_values("chunk_idx").reset_index(drop=True)

        # Build all chunks for this wav while the wav is in memory
        chunk_waveforms: list[torch.Tensor] = []
        chunk_rows: list[pd.Series] = []

        for _, row in group_df.iterrows():
            chunk = slice_waveform_chunk(
                waveform,
                start_s=float(row["chunk_start_s"]),
                end_s=float(row["chunk_end_s"]),
                sample_rate=16000,
            )
            chunk_waveforms.append(chunk)
            chunk_rows.append(row)

        # Free the full wav as soon as chunk tensors are created
        del waveform
        gc.collect()

        # Embed chunks in small batches to reduce Python/GPU overhead
        num_chunks = len(chunk_waveforms)
        print(
            f"[Embed] {split}: wav has {num_chunks} chunks | "
            f"embedding in batches of {EMBED_BATCH_SIZE}"
        )

        for batch_start in range(0, num_chunks, EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, num_chunks)
            batch_waveforms = chunk_waveforms[batch_start:batch_end]
            batch_rows = chunk_rows[batch_start:batch_end]

            result = embedder.embed_batch(batch_waveforms)
            batch_embs = result.embedding

            if isinstance(batch_embs, torch.Tensor):
                batch_embs = batch_embs.detach().cpu().numpy()
            else:
                batch_embs = np.asarray(batch_embs)

            if batch_embs.ndim != 2:
                raise ValueError(
                    f"Expected 2D batch embedding array, got shape {batch_embs.shape} "
                    f"for wav={wav_path}, batch_start={batch_start}"
                )

            if batch_embs.shape[0] != len(batch_rows):
                raise ValueError(
                    f"Batch embedding row mismatch for wav={wav_path}: "
                    f"{batch_embs.shape[0]} embeddings for {len(batch_rows)} rows"
                )

            for row, emb in zip(batch_rows, batch_embs):
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
                embedding_rows.append(emb.astype(np.float32, copy=False))
                label_rows.append(label)

            # Explicitly free temporary batch objects
            del result
            del batch_embs
            del batch_waveforms
            del batch_rows

            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        # Free per-wav chunk buffers before moving on
        del chunk_waveforms
        del chunk_rows
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wavs_in_current_shard += 1
        processed_wav_groups = wav_group_idx + 1

        should_flush = (
            wavs_in_current_shard >= MAX_CONSECUTIVE_FILES
            or len(meta_rows) >= MAX_ROWS_PER_SHARD
        )

        if should_flush:
            flush_shard(
                split_output_dir=split_output_dir,
                shard_idx=shard_idx,
                meta_rows=meta_rows,
                embedding_rows=embedding_rows,
                label_rows=label_rows,
            )

            shard_idx += 1
            progress = {
                "complete": False,
                "next_wav_group_idx": processed_wav_groups,
                "next_shard_idx": shard_idx,
                "processed_wav_groups": processed_wav_groups,
                "total_wav_groups": wavs_in_split,
            }
            save_json(progress_path, progress)

            clear_buffers(meta_rows, embedding_rows, label_rows)
            wavs_in_current_shard = 0

    # Flush any remaining rows
    if meta_rows:
        flush_shard(
            split_output_dir=split_output_dir,
            shard_idx=shard_idx,
            meta_rows=meta_rows,
            embedding_rows=embedding_rows,
            label_rows=label_rows,
        )
        shard_idx += 1
        clear_buffers(meta_rows, embedding_rows, label_rows)

    progress = {
        "complete": True,
        "next_wav_group_idx": wavs_in_split,
        "next_shard_idx": shard_idx,
        "processed_wav_groups": wavs_in_split,
        "total_wav_groups": wavs_in_split,
    }
    save_json(progress_path, progress)

    print(f"[Embed] Split {split} complete.")


# def process_split(
#     split: str,
#     split_path: Path,
#     split_output_dir: Path,
#     embedder: ASTEmbedder,
# ) -> None:
#     if not split_path.exists():
#         raise FileNotFoundError(f"Split parquet not found: {split_path}")
# 
#     print(f"\n[Embed] Loading split: {split_path}")
#     df = pd.read_parquet(split_path)
# 
#     # Deterministic ordering for resumability
#     df = df.sort_values(["wav_path", "chunk_idx"]).reset_index(drop=True)
# 
#     wavs_in_split = df["wav_path"].nunique()
#     print(f"[Embed] Rows in split: {len(df)}")
#     print(f"[Embed] Unique wavs in split: {wavs_in_split}")
#     print(f"[Embed] Label counts:\n{df['queen_present'].value_counts(dropna=False)}")
# 
#     split_output_dir.mkdir(parents=True, exist_ok=True)
#     progress_path = split_output_dir / "progress.json"
# 
#     progress = load_progress(progress_path, wavs_in_split)
# 
#     if progress.get("complete", False):
#         print(f"[Embed] Split {split} already complete. Skipping.")
#         return
# 
#     start_wav_group_idx = int(progress.get("next_wav_group_idx", 0))
#     shard_idx = int(progress.get("next_shard_idx", 0))
# 
#     print(
#         f"[Embed] Resuming split {split} from wav group {start_wav_group_idx}/{wavs_in_split}, "
#         f"next shard index={shard_idx}"
#     )
# 
#     meta_rows: list[dict] = []
#     embedding_rows: list[np.ndarray] = []
#     label_rows: list[int] = []
# 
#     wavs_in_current_shard = 0
# 
#     grouped = df.groupby("wav_path", sort=True)
# 
#     for wav_group_idx, (wav_path, group_df) in enumerate(grouped):
#         if wav_group_idx < start_wav_group_idx:
#             continue
# 
#         print(
#             f"[Embed] {split}: Loading wav {wav_group_idx + 1}/{wavs_in_split}: {wav_path}"
#         )
# 
#         waveform = load_audio_mono_16k(wav_path)
#         group_df = group_df.sort_values("chunk_idx")
# 
#         for _, row in group_df.iterrows():
#             chunk = slice_waveform_chunk(
#                 waveform,
#                 start_s=float(row["chunk_start_s"]),
#                 end_s=float(row["chunk_end_s"]),
#                 sample_rate=16000,
#             )
# 
#             result = embedder.embed(chunk)
#             emb = result.embedding
# 
#             if isinstance(emb, torch.Tensor):
#                 emb = emb.detach().cpu().numpy()
#             else:
#                 emb = np.asarray(emb)
# 
#             emb = np.asarray(emb).squeeze()
# 
#             if emb.ndim != 1:
#                 raise ValueError(
#                     f"Expected 1D embedding vector, got shape {emb.shape} "
#                     f"for wav={wav_path}, chunk_idx={row['chunk_idx']}"
#                 )
# 
#             label = normalize_label(row["queen_present"])
# 
#             meta_rows.append(
#                 {
#                     "dataset_year": row["dataset_year"],
#                     "wav_path": row["wav_path"],
#                     "hive_id": row["hive_id"],
#                     "recording_start_dt": row["recording_start_dt"],
#                     "chunk_idx": row["chunk_idx"],
#                     "chunk_start_s": row["chunk_start_s"],
#                     "chunk_end_s": row["chunk_end_s"],
#                     "chunk_start_dt": row["chunk_start_dt"],
#                     "chunk_end_dt": row["chunk_end_dt"],
#                     "inspection_date": row["inspection_date"],
#                     "days_since_inspection": row["days_since_inspection"],
#                     "queen_present": row["queen_present"],
#                 }
#             )
#             embedding_rows.append(emb)
#             label_rows.append(label)
# 
#         del waveform
#         gc.collect()
# 
#         wavs_in_current_shard += 1
#         processed_wav_groups = wav_group_idx + 1
# 
#         should_flush = (
#             wavs_in_current_shard >= MAX_CONSECUTIVE_FILES
#             or len(meta_rows) >= MAX_ROWS_PER_SHARD
#         )
# 
#         if should_flush:
#             flush_shard(
#                 split_output_dir=split_output_dir,
#                 shard_idx=shard_idx,
#                 meta_rows=meta_rows,
#                 embedding_rows=embedding_rows,
#                 label_rows=label_rows,
#             )
# 
#             shard_idx += 1
#             progress = {
#                 "complete": False,
#                 "next_wav_group_idx": processed_wav_groups,
#                 "next_shard_idx": shard_idx,
#                 "processed_wav_groups": processed_wav_groups,
#                 "total_wav_groups": wavs_in_split,
#             }
#             save_json(progress_path, progress)
# 
#             clear_buffers(meta_rows, embedding_rows, label_rows)
#             wavs_in_current_shard = 0
# 
#     # Flush any remaining rows
#     if meta_rows:
#         flush_shard(
#             split_output_dir=split_output_dir,
#             shard_idx=shard_idx,
#             meta_rows=meta_rows,
#             embedding_rows=embedding_rows,
#             label_rows=label_rows,
#         )
#         shard_idx += 1
#         clear_buffers(meta_rows, embedding_rows, label_rows)
# 
#     progress = {
#         "complete": True,
#         "next_wav_group_idx": wavs_in_split,
#         "next_shard_idx": shard_idx,
#         "processed_wav_groups": wavs_in_split,
#         "total_wav_groups": wavs_in_split,
#     }
#     save_json(progress_path, progress)
# 
#     print(f"[Embed] Split {split} complete.")


def calculate_embeddings() -> None:
    project_root = Path.cwd()
    splits_dir = project_root / "data" / "splits"
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Embed] Initializing AST embedder on device={DEVICE}")
    embedder = ASTEmbedder(device=DEVICE)

    for split in SPLITS:
        split_path = splits_dir / f"{split}.parquet"
        split_output_dir = embeddings_dir / split
        process_split(
            split=split,
            split_path=split_path,
            split_output_dir=split_output_dir,
            embedder=embedder,
        )


if __name__ == "__main__":
    calculate_embeddings()
