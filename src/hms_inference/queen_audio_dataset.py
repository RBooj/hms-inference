from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import ASTFeatureExtractor
from tqdm import tqdm

TARGET_SAMPLE_RATE = 16000


def load_audio_mono(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0).to(torch.float32)


def slice_waveform_chunk(
    waveform_16k: torch.Tensor,
    start_s: float,
    end_s: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> torch.Tensor:
    start_idx = int(round(start_s * sample_rate))
    end_idx = int(round(end_s * sample_rate))

    chunk = waveform_16k[start_idx:end_idx]
    if chunk.numel() == 0:
        raise ValueError(
            f"Empty chunk: start_s={start_s}, end_s={end_s}, "
            f"start_idx={start_idx}, end_idx={end_idx}, waveform_len={waveform_16k.numel()}"
        )
    return chunk


def normalize_label(value) -> float:
    if pd.isna(value):
        raise ValueError("Found missing queen_present label.")
    return float(bool(value))


@dataclass(frozen=True)
class QueenSample:
    waveform: torch.Tensor
    label: torch.Tensor
    days_since_inspection: torch.Tensor
    meta: dict


class QueenAudioDataset(Dataset):
    """
    Dataset backed by queen_train/val/test parquet files.

    Each row describes one chunk of one wav file.
    We load the wav, slice the chunk, and return:
      - waveform: [num_samples]
      - label: scalar float tensor in {0.0, 1.0}
      - days_since_inspection: scalar float tensor
      - meta: dict with identifying fields
    """

    def __init__(
        self,
        parquet_path: str | Path,
        target_sample_rate: int,
        cache_waveforms: bool = False,
    ):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset parquet not found: {self.parquet_path}")

        self.df = pd.read_parquet(self.parquet_path).copy()
        self.df = self.df[self.df["queen_present"].notna()].copy()
        self.df = self.df.sort_values(["wav_path", "chunk_idx"]).reset_index(drop=True)

        self.cache_waveforms = cache_waveforms
        self._waveform_cache: dict[str, torch.Tensor] = {}

        self.target_sample_rate = target_sample_rate

        if len(self.df) == 0:
            raise ValueError(f"No usable rows found in {self.parquet_path}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, wav_path: str) -> torch.Tensor:
        if self.cache_waveforms and wav_path in self._waveform_cache:
            return self._waveform_cache[wav_path]

        wav = load_audio_mono(wav_path, self.target_sample_rate)

        if self.cache_waveforms:
            self._waveform_cache[wav_path] = wav

        return wav

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        wav_path = str(row["wav_path"])
        waveform = self._load_waveform(wav_path)

        chunk = slice_waveform_chunk(
            waveform,
            start_s=float(row["chunk_start_s"]),
            end_s=float(row["chunk_end_s"]),
            sample_rate=self.target_sample_rate,
        )

        label = torch.tensor(normalize_label(row["queen_present"]), dtype=torch.float32)

        days_since = row.get("days_since_inspection", 0.0)
        if pd.isna(days_since):
            days_since = 0.0
        days_since = torch.tensor(float(days_since), dtype=torch.float32)

        meta = {
            "dataset_year": int(row["dataset_year"]),
            "wav_path": wav_path,
            "hive_id": int(row["hive_id"]),
            "chunk_idx": int(row["chunk_idx"]),
            "chunk_start_s": float(row["chunk_start_s"]),
            "chunk_end_s": float(row["chunk_end_s"]),
        }

        return {
            "waveform": chunk,
            "label": label,
            "days_since_inspection": days_since,
            "meta": meta,
        }


def collate_queen_audio(batch: list[dict]) -> dict:
    """
    ASTFeatureExtractor can handle variable-length 1D waveforms in a list,
    so do not pad here.
    """
    waveforms = [item["waveform"] for item in batch]
    labels = torch.stack([item["label"] for item in batch], dim=0)
    days_since = torch.stack([item["days_since_inspection"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]

    return {
        "waveforms": waveforms,
        "labels": labels,
        "days_since_inspection": days_since,
        "metas": metas,
    }


import json


def compute_ast_stats(
    train_loader,
    model_name: str,
    target_sample_rate: int,
    stats_json_path: str | Path | None = None,
    force_recompute: bool = False,
    stats_metadata: dict | None = None,
) -> tuple[float, float]:
    """
    Compute dataset-specific AST input mean/std on the TRAIN split only.

    Important:
    - expects batches produced by collate_queen_audio()
    - therefore batch key is 'waveforms', not 'waveform'
    - feature extractor must run with do_normalize=False here
    """

    stats_json_path = Path(stats_json_path) if stats_json_path is not None else None

    if stats_json_path is not None and stats_json_path.exists() and not force_recompute:
        with stats_json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        mean = float(payload["mean"])
        std = float(payload["std"])

        print(
            f"[Stats] Loaded cached AST stats from {stats_json_path} | "
            f"mean={mean:.6f}, std={std:.6f}"
        )
        return mean, std

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        model_name,
        do_normalize=False,
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for batch in tqdm(train_loader, desc="Computing AST stats"):
        waveforms = batch["waveforms"]  # <-- plural, because this is a collated batch

        for wav in waveforms:
            wav_np = wav.detach().cpu().numpy()

            inputs = feature_extractor(
                wav_np,
                sampling_rate=target_sample_rate,
                return_tensors="pt",
            )

            x = inputs["input_values"]  # shape usually [1, time, mel_bins]

            total_sum += x.sum().item()
            total_sq_sum += (x**2).sum().item()
            total_count += x.numel()

    if total_count == 0:
        raise ValueError("AST stats computation saw zero input elements.")

    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean**2)
    var = max(var, 0.0)
    std = var**0.5

    if stats_json_path is not None:
        stats_json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": model_name,
            "mean": mean,
            "std": std,
            "num_elements": total_count,
            "computed_with_settings": stats_metadata,
        }
        with stats_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"[Stats] Saved AST stats to {stats_json_path} | "
            f"mean={mean:.6f}, std={std:.6f}"
        )

    return mean, std
