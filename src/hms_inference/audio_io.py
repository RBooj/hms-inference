from __future__ import annotations

import torch
import torchaudio


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
