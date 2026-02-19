from dataclasses import dataclass


@dataclass(frozen=True)
class AudioChunk:
    clip_index: int
    start_s: float
    end_s: float


def chunk_30s_to_10s_5overlap() -> list[AudioChunk]:
    # 30s clip makes 5 10 sec chunks with 5s overlap:
    # (0-10), (5-15), (10-20), (15-25), (20-30)
    start_time = [0, 5, 10, 15, 20]
    return [AudioChunk(i, float(s), float(s + 10)) for i, s in enumerate(start_time)]
