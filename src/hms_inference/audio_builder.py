import pandas as pd
from pathlib import Path

from hms_inference.audio_chunk import chunk_30s_to_10s_5overlap
import hms_inference.audio_discovery
from hms_inference.audio_parse import parse_urban_wav_name


def build_chunk_index(wav_paths: list[Path], dataset_year: int) -> pd.DataFrame:
    chunk_plan = chunk_30s_to_10s_5overlap()
    rows = []

    for wav_path in wav_paths:
        meta = parse_urban_wav_name(wav_path)

        for ch in chunk_plan:
            chunk_start_dt = meta.recording_start + pd.Timedelta(seconds=ch.start_s)
            chunk_end_dt = meta.recording_start + pd.Timedelta(seconds=ch.end_s)

            rows.append(
                {
                    "dataset_year": dataset_year,
                    "wav_path": str(wav_path),
                    "hive_id": meta.hive_id,
                    "recording_start_dt": meta.recording_start,
                    "chunk_idx": ch.clip_index,
                    "chunk_start_s": ch.start_s,
                    "chunk_end_s": ch.end_s,
                    "chunk_start_dt": chunk_start_dt.to_pydatetime(),
                    "chunk_end_dt": chunk_end_dt.to_pydatetime(),
                }
            )

    return pd.DataFrame(rows)
