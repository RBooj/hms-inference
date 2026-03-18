from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    clip_index: int
    start_s: float
    end_s: float


def chunk_audio_duration(
    total_duration_s: float, window_s: float = 10.0, hop_s: float = 10.0
) -> list[Chunk]:
    if total_duration_s <= 0:
        return []

    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")

    if hop_s <= 0:
        raise ValueError(f"hop_s must be > 0, got {hop_s}")

    chunks = []
    start_s = 0.0
    clip_index = 0

    while start_s + window_s <= total_duration_s:
        chunks.append(
            Chunk(
                clip_index=clip_index,
                start_s=float(start_s),
                end_s=float(start_s + window_s),
            )
        )
        start_s += hop_s
        clip_index += 1

    return chunks
