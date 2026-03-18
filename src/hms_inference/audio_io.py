from __future__ import annotations

import torch
import torchaudio


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
    # wav: [channels, samples]
    # sr: sample rate int
    wav, sr = torchaudio.load(path)
    print(
        f"[audio_io] loaded: wav.shape={tuple(wav.shape)}, sr={sr}, dtype={wav.dtype}"
    )

    if wav.size(0) > 1:
        # re-encode to mono channel
        wav = torch.mean(wav, dim=0, keepdim=True)
        print(f"[audio_io] encoded to mono: wav.shape={tuple(wav.shape)}")

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        print(f"[audio_io] resampled: wav.shape={tuple(wav.shape)}, sr=16000")

    wav = wav.squeeze(0).to(torch.float32)
    print(
        f"[audio_io] squeezed float32 wav final: shape={tuple(wav.shape)}, dtype={wav.dtype}, "
        f"seconds={wav.numel()/16000:.2f}"
    )
    return wav
