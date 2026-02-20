import pandas as pd
from pathlib import Path
from datetime import timedelta

from hms_inference.audio_chunk import chunk_30s_to_10s_5overlap
from hms_inference.audio_discovery import find_wavs
from hms_inference.audio_parse import parse_urban_wav_name

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "UrBAN" / "data"
AUDIO_ROOT_2021 = DATA_ROOT / "audio" / "beehives_2021"
AUDIO_ROOT_2022 = DATA_ROOT / "audio" / "beehives_2022"

def build_chunk_index(wav_paths: list[Path], dataset_year: int) -> pd.DataFrame:
    chunk_plan = chunk_30s_to_10s_5overlap()
    rows = []

    for wav_path in wav_paths:
        meta = parse_urban_wav_name(wav_path)

        for ch in chunk_plan:
            chunk_start_dt = meta.recording_start + timedelta(seconds=ch.start_s)
            chunk_end_dt = meta.recording_start + timedelta(seconds=ch.end_s)

            rows.append(
                {
                    "dataset_year": dataset_year,
                    "wav_path": str(wav_path),
                    "hive_id": meta.hive_id,
                    "recording_start_dt": meta.recording_start,
                    "chunk_idx": ch.clip_index,
                    "chunk_start_s": ch.start_s,
                    "chunk_end_s": ch.end_s,
                    "chunk_start_dt": chunk_start_dt,
                    "chunk_end_dt": chunk_end_dt,
                }
            )

    return pd.DataFrame(rows)

test_wavs = find_wavs(AUDIO_ROOT_2021)
df = build_chunk_index(test_wavs[:20], dataset_year=2021)
print(df.head())
print("rows:" , len(df), "expected:", len(test_wavs[:20])*5)
