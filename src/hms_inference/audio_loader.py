import pandas as pd
import re
import torch
import torchaudio

from torchcodec.decoders import AudioDecoder
from pathlib import Path
from datetime import timedelta, datetime
from dataclasses import dataclass
from __future__ import annotations

# Constants for chunking stradegy
DEFAULT_CHUNK_LENGTH = 10.0
DEFAULT_HOP_LENGTH = 10.0

# regex parsing for wav filenames
FILENAME_RE = re.compile(
    r"""
        (?P<date>\d{2}-\d{2}-\d{4})         # dd-mm-yyy
        _
        (?P<hour>\d{2})h(?P<minute>\d{2})   # HHhMM
        _
        (?P<hive>hive-\d+)                  # hive-####
        \.(wav)                             # extention
        $
        """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class Chunk:
    clip_index: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class AudioMeta:
    hive_id: int
    recording_start_dt: datetime


def build_chunk_df(
    wav_paths: list[Path],
    dataset_year: int,
    *,
    chunk_length_s: float = DEFAULT_CHUNK_LENGTH,
    hop_length_s: float = DEFAULT_HOP_LENGTH,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns describing chunks of audio
    and rows representing chunks
    """
    rows = []

    for wav_path in wav_paths:
        wav_meta = parse_urban_wav_name(wav_path)
        wav_chunks = chunk_audio_duration(wav_path, chunk_length_s, hop_length_s)

        for chunk in wav_chunks:
            chunk_start_dt = wav_meta.recording_start_dt + timedelta(
                seconds=chunk.start_s
            )
            chunk_end_dt = wav_meta.recording_start_dt + timedelta(seconds=chunk.end_s)

            # construct rows of parsed chunks
            rows.append(
                {
                    "dataset_year": dataset_year,
                    "wav_path": str(wav_path),
                    "hive_id": wav_meta.hive_id,
                    "recording_start_dt": wav_meta.recording_start_dt,
                    "chunk_idx": chunk.clip_index,
                    "chunk_start_s": chunk.start_s,
                    "chunk_end_s": chunk.end_s,
                    "chunk_start_dt": pd.to_datetime(chunk_start_dt, utc=True),
                    "chunk_end_dt": pd.to_datetime(chunk_end_dt, utc=True),
                }
            )

    chunk_df = pd.DataFrame(rows)
    chunk_df["hive_id"] = pd.to_numeric(chunk_df["hive_id"], errors="coerce").astype(
        "Int64"
    )
    return chunk_df


def discover_wav_files(audio_root: Path) -> list[Path]:
    """
    Returns a list of wav files given a path to
    directory containing wav files
    """
    wavs = []
    for ext in ("*.wav", "*.WAV"):
        wavs.extend(audio_root.rglob(ext))
    return sorted(wavs)


def parse_urban_wav_name(wav_path: Path) -> AudioMeta:
    """
    given a path of one wav file, turn its filename into a
    datetime object and hive id
    """
    m = FILENAME_RE.search(wav_path.name)
    if not m:
        raise ValueError(f"Unrecognized wav filename: {wav_path.name}")

    wav_date = m.group("date")
    wav_hour = int(m.group("hour"))
    wav_minute = int(m.group("minute"))
    wav_hive = int(m.group("hive").split("-")[1])

    wav_start_dt = datetime.strptime(wav_date, "%d-%m-%Y").replace(
        hour=wav_hour, minute=wav_minute
    )
    return AudioMeta(hive_id=wav_hive, recording_start_dt=wav_start_dt)


def get_audio_duration_s(wav_path: Path) -> float:
    decoder = AudioDecoder(str(wav_path))
    return decoder.metadata.duration_seconds


def chunk_audio_duration(
    wav_path: Path,
    chunk_length_s: float = DEFAULT_CHUNK_LENGTH,
    hop_length_s: float = DEFAULT_HOP_LENGTH,
) -> list[Chunk]:
    """
    For a given audio file, returns the start/stop times of all the chunks it will produce
    """
    total_length_s = get_audio_duration_s(wav_path)

    if total_length_s <= 0:
        return []
    if chunk_length_s <= 0:
        raise ValueError(f"chunk_length_s must be larger than 0, got {chunk_length_s}")
    if hop_length_s <= 0:
        raise ValueError(f"hop_length_s must be > 0, got {hop_length_s}")

    chunks = []
    start_s = 0.0
    clip_index = 0

    while start_s + chunk_length_s <= total_length_s:
        chunks.append(
            Chunk(
                clip_index=clip_index,
                start_s=float(start_s),
                end_s=float(start_s + chunk_length_s),
            )
        )
        start_s += hop_length_s
        clip_index += 1

    return chunks
