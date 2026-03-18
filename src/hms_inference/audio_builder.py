import pandas as pd
from torchcodec.decoders import AudioDecoder

from pathlib import Path
from datetime import timedelta

from hms_inference.audio_chunk import chunk_audio_duration
from hms_inference.audio_parse import parse_urban_wav_name

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "UrBAN" / "data"
AUDIO_ROOT_2021 = DATA_ROOT / "audio" / "beehives_2021"
AUDIO_ROOT_2022 = DATA_ROOT / "audio" / "beehives_2022"


def get_audio_duration_seconds(wav_path: Path) -> float:
    decoder = AudioDecoder(str(wav_path))
    return decoder.metadata.duration_seconds


def build_chunk_index(
    wav_paths: list[Path],
    dataset_year: int,
    *,
    window_s: float = 10.0,
    hop_s: float = 10.0,
) -> pd.DataFrame:
    rows = []

    for wav_path in wav_paths:
        meta = parse_urban_wav_name(wav_path)
        duration_s = get_audio_duration_seconds(wav_path)
        chunk_plan = chunk_audio_duration(
            total_duration_s=duration_s, window_s=window_s, hop_s=hop_s
        )

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

    chunk = pd.DataFrame(rows)
    chunk["hive_id"] = pd.to_numeric(chunk["hive_id"], errors="coerce").astype("Int64")
    return chunk


if __name__ == "__main__":
    from hms_inference.audio_discovery import find_wavs

    test_wavs = find_wavs(AUDIO_ROOT_2021)
    df = build_chunk_index(test_wavs[:3], dataset_year=2021, window_s=10.0, hop_s=10.0)

    print(df.head(15).to_string(index=False))
    print("\nrows:", len(df))

    print("\nChunks per wav:")
    print(df.groupby("wav_path")["chunk_idx"].count())
